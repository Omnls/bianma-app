//! Provider 绑定事实采集与静态解析
//!
//! 本模块实现“单次选择事实采集 + 静态 Provider 绑定事实解析”切片：
//! - 在 `ProviderRouter` 同一次配置/队列/熔断读取内为每个候选生成有序事实；
//! - 事实只含白名单字段，绝不包含 `settings_config` / `meta` 原文或任何认证值；
//! - 对常规静态 Claude Provider，复用现有 Adapter 提取与
//!   `ModelMapping` 优先级，解析真实 base URL、协议与实际上游模型；
//! - 请求期才能确定或可能动态变化的绑定（托管账号、Copilot 动态端点、
//!   is_full_url、endpoint_auto_select、不安全 URL、无法证明的 Codex 操作等）
//!   一律返回封闭的 `IncomparableReason`，绝不猜测。
//!
//! 本切片不生成 Site/Endpoint/Deployment/AccountSelector 数字 ID，不生成指纹，
//! 不做请求期持久化，也不参与影子比较；这些由后续切片在配置发布时消费本模块
//! 输出的事实完成。

use crate::provider::{AuthBindingSource, Provider};
use crate::proxy::model_mapper::ModelMapping;
use crate::proxy::providers::{
    claude_api_format_needs_transform, get_claude_api_format, ClaudeAdapter, ProviderAdapter,
};
use serde::Serialize;

/// 可比较候选容量上限（与 routing-core “完整候选 > 16 拒绝”规则对齐）。
///
/// 超过上限时整次比较被封闭为 [`IncomparableReason::CandidateOverflow`]，
/// 但不改变旧 `select_providers` 返回的发送列表。
pub const MAX_COMPARABLE_CANDIDATES: usize = 16;

/// 单次选择所处的模式（故障转移开关状态）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    FailoverEnabled,
    FailoverDisabled,
}

/// 候选在旧选择语义下的准入状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionState {
    /// 进入旧 providers 返回列表
    Admitted,
    /// 熔断器 Open 且未到恢复超时，未纳入发送列表
    RejectedCircuitOpen,
    /// 故障转移队列引用但 providers 表中不存在（旧语义静默跳过）
    SkippedMissingProvider,
}

/// 故障转移关闭时“当前供应商”ID 的来源
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CurrentProviderSource {
    /// `get_effective_current_provider` 返回的有效 current：
    /// 不区分有效本机 settings 或 helper 内部 DB 回退
    EffectiveCurrent,
    /// 数据库 is_current 回退
    DatabaseCurrent,
}

/// 本机认证来源（有类型，不携带任何认证值）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthSourceKind {
    /// 认证材料来自 Provider 自身配置（本机静态可读）
    ProviderConfig,
    /// 托管账号认证（含旧 Copilot 兼容字段 githubAccountId），token 在请求期动态获取
    ManagedAccount,
}

/// wire 协议 / 操作
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    /// Anthropic Messages 原生透传
    AnthropicMessages,
    /// OpenAI Chat Completions
    OpenAiChatCompletions,
    /// OpenAI Responses
    OpenAiResponses,
}

/// 协议/操作事实
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProtocolFact {
    pub wire: WireProtocol,
    /// 发送前是否发生请求体格式变换（对照 forwarder 的 needs_transform 路径：
    /// Claude 的 openai_chat/openai_responses 会做 Anthropic→OpenAI 变换，因此这里
    /// 只承诺协议与“是否变换”，不承诺变换后的最终端点路径）
    pub request_transform: bool,
}

/// 规范端点事实
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EndpointFact {
    /// base_url 的静态提取来源（复用的 Adapter 提取函数）
    pub extractor: &'static str,
    /// 规范化 base URL：http/https、无 userinfo/query/fragment、无尾斜杠
    pub base_url: String,
}

/// 具体模型事实（请求的实际上游模型）
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelFact {
    /// 客户端请求模型
    pub requested_model: String,
    /// 请求是否启用 thinking（由 model_mapper::has_thinking_enabled 判定）
    pub thinking_enabled: bool,
    /// ModelMapping 规则是否改变了模型名
    pub mapping_applied: bool,
    /// 应用 ModelMapping 优先级后的实际上游模型
    pub upstream_model: String,
}

/// 已解析的静态绑定事实
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedBinding {
    pub endpoint: EndpointFact,
    pub protocol: ProtocolFact,
    pub model: ModelFact,
    /// 本机认证来源（ManagedAccount 已在解析早期被封闭，此处实际恒为 ProviderConfig）
    pub auth_source: AuthSourceKind,
}

/// 封闭的不可比较原因：凡是请求期才能确定、可能动态变化或不安全的绑定
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum IncomparableReason {
    /// 托管账号认证（含旧 Copilot 兼容字段），账号与 token 在请求期动态确定
    ManagedAccountAuth,
    /// Copilot 动态端点：API endpoint 由 CopilotAuthManager 在请求期提供
    CopilotDynamicEndpoint,
    /// meta.is_full_url：base_url 被当作完整端点，实际请求地址由客户端路径决定
    FullUrlEndpoint,
    /// meta.endpoint_auto_select：端点随测速结果动态变化
    EndpointAutoSelect,
    /// 不支持或可能动态变化的 wire API 格式 / 应用
    UnsupportedWireApi { detail: String },
    /// 无法从配置静态提取 base_url
    BaseUrlUnavailable,
    /// base_url 不安全：含 userinfo/query/fragment 或非 http/https
    UnsafeBaseUrl,
    /// 请求模型缺失或为占位值（"unknown"）
    MissingRequestModel,
    /// 映射后的上游模型为空/无效
    EmptyUpstreamModel,
    /// Codex chat/responses 操作在当前接缝无法由事实证明
    OperationUndetermined,
    /// 队列引用的 Provider 在数据库中不存在
    ProviderNotFound,
    /// 完整候选数超过可比较容量（routing-core > 16 拒绝）
    CandidateOverflow { count: usize, max: usize },
    /// 至少一个候选绑定不可比较（记录其有序序号）；整次选择随之封闭，
    /// `selection_incomparable == None` 仅表示全量候选绑定均可证明
    CandidateBindingsIncomparable { ordinals: Vec<usize> },
}

/// 单个候选的绑定事实解析结果
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", content = "fact", rename_all = "snake_case")]
pub enum BindingFact {
    Resolved(ResolvedBinding),
    Incomparable(IncomparableReason),
}

/// 模型映射规则轮廓（仅模型名，不含 Secret）。
///
/// 注意：这只是配置规则的白名单轮廓，供下一次配置发布时持久化与比较，
/// 本身不是也不冒充 ModelDeployment（不携带任何数字 ID / 指纹）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ModelMappingOutline {
    pub haiku_model: Option<String>,
    pub sonnet_model: Option<String>,
    pub opus_model: Option<String>,
    pub default_model: Option<String>,
    pub reasoning_model: Option<String>,
}

impl From<&ModelMapping> for ModelMappingOutline {
    fn from(mapping: &ModelMapping) -> Self {
        Self {
            haiku_model: mapping.haiku_model.clone(),
            sonnet_model: mapping.sonnet_model.clone(),
            opus_model: mapping.opus_model.clone(),
            default_model: mapping.default_model.clone(),
            reasoning_model: mapping.reasoning_model.clone(),
        }
    }
}

/// 不含原始 settings_config / meta / 认证值的白名单 Provider 描述
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderSnapshot {
    pub provider_id: String,
    pub name: String,
    pub in_failover_queue: bool,
    pub sort_index: Option<usize>,
    /// meta.provider_type 标记（如 "github_copilot"）
    pub provider_type: Option<String>,
    /// meta.api_format 原文标记（Claude 静态 API 格式）
    pub api_format: Option<String>,
    pub is_full_url: bool,
    pub endpoint_auto_select: bool,
    pub auth_source: AuthSourceKind,
    pub model_mapping: ModelMappingOutline,
    /// 自定义端点数量（端点 URL 可能内嵌参数，不进入白名单，只记录数量）
    pub custom_endpoint_count: usize,
    /// 是否配置用量脚本（脚本可能含 api_key，只记录存在性）
    pub has_usage_script: bool,
    /// 是否配置供应商级代理（代理配置可能含口令，只记录存在性）
    pub has_proxy_config: bool,
    pub live_config_managed: Option<bool>,
}

/// 单个有序候选的完整事实
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateFact {
    /// 在有序候选中的位置（0 起，与旧选择遍历顺序一致）
    pub ordinal: usize,
    pub provider_id: String,
    pub provider_name: String,
    /// 旧选择语义下的准入状态
    pub admission: AdmissionState,
    /// 本次选择是否实际读取了熔断器（故障转移关闭、候选缺失等路径未读取）
    pub breaker_checked: bool,
    /// 白名单快照；队列引用缺失时为 None
    pub snapshot: Option<ProviderSnapshot>,
    pub binding: BindingFact,
}

/// 一次选择的完整有序事实
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionFacts {
    pub app_type: String,
    pub mode: SelectionMode,
    /// 仅故障转移关闭时有意义：当前供应商 ID 的来源
    pub current_provider_source: Option<CurrentProviderSource>,
    /// 全量有序候选事实（含未 admitted 的候选；不因绑定失败/cooldown 删除）
    pub candidates: Vec<CandidateFact>,
    /// 整次不可比较的封闭原因（候选超限或任一候选绑定不可比）；
    /// None 仅表示全量候选绑定均可证明（绝不是只比较了子集）
    pub selection_incomparable: Option<IncomparableReason>,
}

impl SelectionFacts {
    pub fn new(app_type: &str, mode: SelectionMode) -> Self {
        Self {
            app_type: app_type.to_string(),
            mode,
            current_provider_source: None,
            candidates: Vec::new(),
            selection_incomparable: None,
        }
    }

    /// 完成采集后收口整次可比较性（不改变任何发送行为）：
    /// - 候选数超过可比较容量 → CandidateOverflow（容量原因优先）；
    /// - 否则任一候选绑定不可比 → CandidateBindingsIncomparable，记录其序号；
    /// - 两者都没有 → None，表示全量候选绑定均可证明。
    pub(crate) fn mark_capacity(&mut self) {
        if self.candidates.len() > MAX_COMPARABLE_CANDIDATES {
            self.selection_incomparable = Some(IncomparableReason::CandidateOverflow {
                count: self.candidates.len(),
                max: MAX_COMPARABLE_CANDIDATES,
            });
            return;
        }
        let ordinals: Vec<usize> = self
            .candidates
            .iter()
            .filter(|c| matches!(c.binding, BindingFact::Incomparable(_)))
            .map(|c| c.ordinal)
            .collect();
        if !ordinals.is_empty() {
            self.selection_incomparable =
                Some(IncomparableReason::CandidateBindingsIncomparable { ordinals });
        }
    }
}

/// `select_providers_with_facts` 的返回值
#[derive(Debug)]
pub struct SelectWithFacts {
    /// 与旧 `select_providers` 完全一致的有序 Provider 列表（发送路径专用）
    pub providers: Vec<Provider>,
    /// 同一次选择内采集的有序绑定事实
    pub facts: SelectionFacts,
}

/// 请求期已知事实（由 RequestContext 在已解析请求上构造）
#[derive(Debug)]
pub struct RequestFacts<'a> {
    /// 请求中的模型名（handler 已提取；缺省时为 "unknown"）
    pub model: &'a str,
    /// 请求是否启用 thinking
    pub thinking_enabled: bool,
}

/// 为队列引用但数据库中不存在的候选生成占位事实（保留有序性，便于后续对账）
pub fn capture_missing_candidate(ordinal: usize, provider_id: &str) -> CandidateFact {
    CandidateFact {
        ordinal,
        provider_id: provider_id.to_string(),
        provider_name: String::new(),
        admission: AdmissionState::SkippedMissingProvider,
        breaker_checked: false,
        snapshot: None,
        binding: BindingFact::Incomparable(IncomparableReason::ProviderNotFound),
    }
}

/// 从已经读取的 Provider 捕获白名单快照并解析绑定事实。
///
/// 纯函数：不访问数据库、不访问熔断器、不读取环境变量，永不失败——
/// 任何无法静态证明的绑定都落入封闭的 `Incomparable` 结果，
/// 因此事实解析绝不可能导致原本可发送的请求失败。
pub fn capture_candidate(
    ordinal: usize,
    provider: &Provider,
    admission: AdmissionState,
    breaker_checked: bool,
    app_type: &str,
    request: &RequestFacts<'_>,
) -> CandidateFact {
    let snapshot = snapshot_provider(provider);
    let binding = resolve_binding(app_type, provider, &snapshot, request);
    CandidateFact {
        ordinal,
        provider_id: provider.id.clone(),
        provider_name: provider.name.clone(),
        admission,
        breaker_checked,
        snapshot: Some(snapshot),
        binding,
    }
}

/// 生成白名单快照（不含 settings_config / meta 原文与任何认证值）
fn snapshot_provider(provider: &Provider) -> ProviderSnapshot {
    let meta = provider.meta.as_ref();
    let mapping = ModelMapping::from_provider(provider);
    ProviderSnapshot {
        provider_id: provider.id.clone(),
        name: provider.name.clone(),
        in_failover_queue: provider.in_failover_queue,
        sort_index: provider.sort_index,
        provider_type: meta.and_then(|m| m.provider_type.clone()),
        api_format: meta.and_then(|m| m.api_format.clone()),
        is_full_url: meta.and_then(|m| m.is_full_url).unwrap_or(false),
        endpoint_auto_select: meta.and_then(|m| m.endpoint_auto_select).unwrap_or(false),
        auth_source: detect_auth_source(provider),
        model_mapping: ModelMappingOutline::from(&mapping),
        custom_endpoint_count: meta.map(|m| m.custom_endpoints.len()).unwrap_or(0),
        has_usage_script: meta.is_some_and(|m| m.usage_script.is_some()),
        has_proxy_config: meta.is_some_and(|m| m.proxy_config.is_some()),
        live_config_managed: meta.and_then(|m| m.live_config_managed),
    }
}

/// 判定本机认证来源：通用 authBinding 优先，旧 Copilot 兼容字段同样视为托管账号
fn detect_auth_source(provider: &Provider) -> AuthSourceKind {
    if let Some(meta) = provider.meta.as_ref() {
        if let Some(binding) = meta.auth_binding.as_ref() {
            if binding.source == AuthBindingSource::ManagedAccount {
                return AuthSourceKind::ManagedAccount;
            }
        }
        // 旧 Copilot 兼容字段：githubAccountId 存在即视为托管账号认证
        if meta.github_account_id.is_some() {
            return AuthSourceKind::ManagedAccount;
        }
    }
    AuthSourceKind::ProviderConfig
}

/// 静态绑定解析：常规静态 Claude / Codex + ProviderConfig 可成功；
/// 其余一律返回封闭的 IncomparableReason
fn resolve_binding(
    app_type: &str,
    provider: &Provider,
    snapshot: &ProviderSnapshot,
    request: &RequestFacts<'_>,
) -> BindingFact {
    use IncomparableReason as R;

    // 1) 托管账号认证（含旧 Copilot 兼容字段）：账号与 token 在请求期动态确定
    if snapshot.auth_source == AuthSourceKind::ManagedAccount {
        return BindingFact::Incomparable(R::ManagedAccountAuth);
    }

    // 2) Copilot 动态端点（providerType 标记）：endpoint 由 CopilotAuthManager 请求期提供
    if snapshot.provider_type.as_deref() == Some("github_copilot") {
        return BindingFact::Incomparable(R::CopilotDynamicEndpoint);
    }

    // 3) is_full_url / endpoint_auto_select：实际请求地址无法由静态配置证明
    if snapshot.is_full_url {
        return BindingFact::Incomparable(R::FullUrlEndpoint);
    }
    if snapshot.endpoint_auto_select {
        return BindingFact::Incomparable(R::EndpointAutoSelect);
    }

    // 4) 仅支持常规静态 Claude / Codex；其余应用或可能动态变化的 wire API 一律封闭
    let (extractor, base_url_raw) = match app_type {
        "claude" => match ClaudeAdapter::new().extract_base_url(provider) {
            Ok(url) => ("ClaudeAdapter::extract_base_url", url),
            Err(_) => return BindingFact::Incomparable(R::BaseUrlUnavailable),
        },
        // Codex 操作在当前接缝无法由事实证明：不再静态提取 base_url，直接封闭不猜
        "codex" => return BindingFact::Incomparable(R::OperationUndetermined),
        other => {
            return BindingFact::Incomparable(R::UnsupportedWireApi {
                detail: format!("应用 {other} 的 wire API 不支持静态绑定解析或可能动态变化"),
            });
        }
    };

    // base_url 指向 Copilot 动态域（兼容无 providerType 标记的旧数据）
    if base_url_raw.contains("githubcopilot.com") {
        return BindingFact::Incomparable(R::CopilotDynamicEndpoint);
    }

    // 5) URL 安全校验：http/https、无 userinfo/query/fragment
    let base_url = match normalize_safe_base_url(&base_url_raw) {
        Ok(url) => url,
        Err(reason) => return BindingFact::Incomparable(reason),
    };

    // 6) 请求模型与实际上游模型（复用 ModelMapping 优先级）：
    //    原始 request.model 原样进入 map_model，与真实 forwarder 一致；
    //    trim 仅用于判定纯空白/占位，不改写进入映射与事实的原文
    let requested_model = request.model;
    let trimmed_model = requested_model.trim();
    if trimmed_model.is_empty() || trimmed_model == "unknown" {
        return BindingFact::Incomparable(R::MissingRequestModel);
    }
    let mapping = ModelMapping::from_provider(provider);
    let upstream_model = mapping.map_model(requested_model, request.thinking_enabled);
    // 有效上游模型必须非空且不能是占位 unknown（unknown 不能宣称具体模型）
    let trimmed_upstream = upstream_model.trim();
    if trimmed_upstream.is_empty() || trimmed_upstream == "unknown" {
        return BindingFact::Incomparable(R::EmptyUpstreamModel);
    }

    // 7) 协议/操作：
    //    - Claude 对照 forwarder 非 Copilot 路径的 get_claude_api_format
    //      （Copilot 已在步骤 1/2 封闭；openai_chat/openai_responses 会触发
    //      Anthropic→OpenAI 变换，因此只承诺协议与 request_transform 标记）；
    //    - Codex 已在步骤 4 封闭（当前接缝无法由事实证明其 chat/responses 操作）。
    let protocol = match app_type {
        "claude" => {
            // 显式配置的未知 api_format 不能冒用 helper 默认 fallback 宣称支持：
            // 按 helper 真实优先级（meta.api_format > settings_config.api_format）
            // 核对，未知值一律封闭为 UnsupportedWireApi，且不回显未知原值
            let explicit_format = provider
                .meta
                .as_ref()
                .and_then(|m| m.api_format.as_deref())
                .or_else(|| {
                    provider
                        .settings_config
                        .get("api_format")
                        .and_then(|v| v.as_str())
                });
            if let Some(format) = explicit_format {
                if !matches!(format, "anthropic" | "openai_chat" | "openai_responses") {
                    return BindingFact::Incomparable(R::UnsupportedWireApi {
                        detail:
                            "Claude Provider 配置了无法识别的 api_format，无法静态证明 wire API"
                                .to_string(),
                    });
                }
            }
            let api_format = get_claude_api_format(provider);
            let request_transform = claude_api_format_needs_transform(api_format);
            let wire = match api_format {
                "openai_chat" => WireProtocol::OpenAiChatCompletions,
                "openai_responses" => WireProtocol::OpenAiResponses,
                _ => WireProtocol::AnthropicMessages,
            };
            ProtocolFact {
                wire,
                request_transform,
            }
        }
        // 步骤 4 已封闭 codex；此分支仅保证穷尽性，同样保守不可比、绝不猜测
        "codex" => return BindingFact::Incomparable(R::OperationUndetermined),
        // 步骤 4 已保证只会是 claude / codex
        _ => unreachable!("unsupported app 已在步骤 4 封闭"),
    };

    BindingFact::Resolved(ResolvedBinding {
        endpoint: EndpointFact {
            extractor,
            base_url,
        },
        protocol,
        model: ModelFact {
            requested_model: requested_model.to_string(),
            thinking_enabled: request.thinking_enabled,
            mapping_applied: upstream_model != requested_model,
            upstream_model,
        },
        auth_source: snapshot.auth_source,
    })
}

/// 规范化并校验 base URL：仅允许 http/https，拒绝 userinfo/query/fragment。
/// 使用已依赖的 url::Url 真实解析（不手工拆 authority）；返回的规范形式
/// 必须与真实发送值（仅去尾斜杠）逐字节一致，无法证明等价时保守拒绝
fn normalize_safe_base_url(raw: &str) -> Result<String, IncomparableReason> {
    use IncomparableReason as R;
    // 首尾及内部 ASCII 空白/控制字符一律拒绝（非法端口由 Url::parse 拒绝）
    if raw
        .chars()
        .any(|c| c.is_ascii_whitespace() || c.is_ascii_control())
    {
        return Err(R::UnsafeBaseUrl);
    }
    let url = url::Url::parse(raw).map_err(|_| R::UnsafeBaseUrl)?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(R::UnsafeBaseUrl);
    }
    if url.host_str().is_none_or(|h| h.is_empty())
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(R::UnsafeBaseUrl);
    }
    let stripped = raw.trim_end_matches('/');
    let canonical = url.as_str().trim_end_matches('/');
    // 无法证明解析规范形与真实发送值等价时，保守封闭
    if canonical != stripped {
        return Err(R::UnsafeBaseUrl);
    }
    Ok(stripped.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{AuthBinding, ProviderMeta};
    use serde_json::json;

    fn claude_provider(settings: serde_json::Value) -> Provider {
        Provider::with_id(
            "fake-claude-1".to_string(),
            "Fake Claude".to_string(),
            settings,
            None,
        )
    }

    fn codex_provider(settings: serde_json::Value) -> Provider {
        Provider::with_id(
            "fake-codex-1".to_string(),
            "Fake Codex".to_string(),
            settings,
            None,
        )
    }

    fn request(model: &str, thinking: bool) -> RequestFacts<'_> {
        RequestFacts {
            model,
            thinking_enabled: thinking,
        }
    }

    #[test]
    fn blank_model_rejected_but_whitespace_model_kept_verbatim() {
        let provider = claude_provider(json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://claude-fake.example.test",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "fake-upstream-sonnet"
            }
        }));
        let req = request("  claude-sonnet-4-5-fake  ", false);

        let fact = capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req);
        match &fact.binding {
            BindingFact::Resolved(r) => {
                // trim 仅用于判空白；进入映射与事实的原文保持原样
                assert_eq!(r.model.requested_model, "  claude-sonnet-4-5-fake  ");
                assert_eq!(r.model.upstream_model, "fake-upstream-sonnet");
            }
            other => panic!("expected resolved binding, got {other:?}"),
        }

        // 纯空白模型 → 封闭
        let req_blank = request("   ", false);
        assert!(matches!(
            capture_candidate(
                0,
                &provider,
                AdmissionState::Admitted,
                true,
                "claude",
                &req_blank
            )
            .binding,
            BindingFact::Incomparable(IncomparableReason::MissingRequestModel)
        ));
    }

    #[test]
    fn unknown_explicit_api_format_is_rejected() {
        let req = request("claude-sonnet-4-5-fake", false);

        let mut provider = claude_provider(
            json!({"env": {"ANTHROPIC_BASE_URL": "https://claude-fake.example.test"}}),
        );
        provider.meta = Some(ProviderMeta {
            api_format: Some("grpc-fake-unknown".to_string()),
            ..Default::default()
        });

        let fact = capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req);
        match &fact.binding {
            BindingFact::Incomparable(IncomparableReason::UnsupportedWireApi { detail }) => {
                // 未知原值不得回显
                assert!(!detail.contains("grpc-fake-unknown"));
            }
            other => panic!("expected UnsupportedWireApi, got {other:?}"),
        }
    }

    #[test]
    fn resolves_static_claude_provider_without_leaking_secrets() {
        let provider = claude_provider(json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://claude-fake.example.test/",
                "ANTHROPIC_AUTH_TOKEN": "sk-fake-TEST-ONLY-DO-NOT-USE",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "fake-upstream-sonnet"
            }
        }));
        let req = request("claude-sonnet-4-5-fake", false);

        let fact = capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req);
        let resolved = match &fact.binding {
            BindingFact::Resolved(r) => r,
            other => panic!("expected resolved binding, got {other:?}"),
        };

        assert_eq!(
            resolved.endpoint.base_url,
            "https://claude-fake.example.test"
        );
        assert_eq!(resolved.protocol.wire, WireProtocol::AnthropicMessages);
        assert!(!resolved.protocol.request_transform);
        assert_eq!(resolved.model.upstream_model, "fake-upstream-sonnet");
        assert!(resolved.model.mapping_applied);
        assert_eq!(resolved.auth_source, AuthSourceKind::ProviderConfig);

        // 安全边界：序列化后的事实不得包含任何认证值或 settings_config 原文
        let serialized = serde_json::to_string(&fact).unwrap();
        assert!(!serialized.contains("sk-fake-TEST-ONLY-DO-NOT-USE"));
        assert!(!serialized.contains("ANTHROPIC_AUTH_TOKEN"));
    }

    #[test]
    fn model_mapping_prefers_reasoning_model_when_thinking_enabled() {
        let provider = claude_provider(json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://claude-fake.example.test",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "fake-upstream-sonnet",
                "ANTHROPIC_REASONING_MODEL": "fake-upstream-reasoning"
            }
        }));
        let req = request("claude-sonnet-4-5-fake", true);

        let fact = capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req);
        match &fact.binding {
            BindingFact::Resolved(r) => {
                assert_eq!(r.model.upstream_model, "fake-upstream-reasoning");
                assert!(r.model.thinking_enabled);
            }
            other => panic!("expected resolved binding, got {other:?}"),
        }
    }

    #[test]
    fn closed_incomparable_reasons_cover_dynamic_bindings() {
        let req = request("claude-sonnet-4-5-fake", false);
        let base = || {
            claude_provider(
                json!({"env": {"ANTHROPIC_BASE_URL": "https://claude-fake.example.test"}}),
            )
        };

        // 旧 Copilot 兼容字段 → 托管账号认证
        let mut p = base();
        p.meta = Some(ProviderMeta {
            github_account_id: Some("fake-account".to_string()),
            ..Default::default()
        });
        assert!(matches!(
            capture_candidate(0, &p, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::ManagedAccountAuth)
        ));

        // 通用托管账号绑定 → 托管账号认证
        let mut p = base();
        p.meta = Some(ProviderMeta {
            auth_binding: Some(AuthBinding {
                source: AuthBindingSource::ManagedAccount,
                auth_provider: Some("fake_idp".to_string()),
                account_id: None,
            }),
            ..Default::default()
        });
        assert!(matches!(
            capture_candidate(0, &p, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::ManagedAccountAuth)
        ));

        // Copilot providerType 标记（无托管字段）→ 动态端点
        let mut p = base();
        p.meta = Some(ProviderMeta {
            provider_type: Some("github_copilot".to_string()),
            ..Default::default()
        });
        assert!(matches!(
            capture_candidate(0, &p, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::CopilotDynamicEndpoint)
        ));

        // is_full_url
        let mut p = base();
        p.meta = Some(ProviderMeta {
            is_full_url: Some(true),
            ..Default::default()
        });
        assert!(matches!(
            capture_candidate(0, &p, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::FullUrlEndpoint)
        ));

        // endpoint_auto_select
        let mut p = base();
        p.meta = Some(ProviderMeta {
            endpoint_auto_select: Some(true),
            ..Default::default()
        });
        assert!(matches!(
            capture_candidate(0, &p, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::EndpointAutoSelect)
        ));

        // 请求模型缺失/占位 → 封闭
        let req_unknown = request("unknown", false);
        assert!(matches!(
            capture_candidate(
                0,
                &base(),
                AdmissionState::Admitted,
                true,
                "claude",
                &req_unknown
            )
            .binding,
            BindingFact::Incomparable(IncomparableReason::MissingRequestModel)
        ));

        // 不支持的应用 → 封闭
        assert!(matches!(
            capture_candidate(0, &base(), AdmissionState::Admitted, true, "gemini", &req).binding,
            BindingFact::Incomparable(IncomparableReason::UnsupportedWireApi { .. })
        ));
    }

    #[test]
    fn mapped_upstream_unknown_is_incomparable() {
        let provider = claude_provider(json!({
            "env": {
                "ANTHROPIC_BASE_URL": "https://claude-fake.example.test",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": "unknown"
            }
        }));
        let req = request("claude-sonnet-4-5-fake", false);

        assert!(matches!(
            capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req).binding,
            BindingFact::Incomparable(IncomparableReason::EmptyUpstreamModel)
        ));
    }

    #[test]
    fn unsafe_base_urls_are_rejected() {
        let req = request("claude-sonnet-4-5-fake", false);

        for url in [
            "https://user:password@evil-fake.example.test",
            "https://fake.example.test/path?key=value",
            "https://fake.example.test/path#frag",
            "ftp://fake.example.test",
            " https://fake.example.test",
            "https://fake.example.test ",
            "https://fake.example.test/a b",
            "https://fake.example.test:99999",
            "https://",
        ] {
            let provider = claude_provider(json!({"env": {"ANTHROPIC_BASE_URL": url}}));
            assert!(
                matches!(
                    capture_candidate(0, &provider, AdmissionState::Admitted, true, "claude", &req)
                        .binding,
                    BindingFact::Incomparable(IncomparableReason::UnsafeBaseUrl)
                ),
                "url 应被拒绝: {url}"
            );
        }
    }

    #[test]
    fn codex_binding_is_always_operation_undetermined() {
        let provider = codex_provider(json!({"base_url": "https://codex-fake.example.test"}));

        // input / messages / 双字段 / 空 body：当前接缝一律无法证明操作，封闭不猜
        for body in [
            json!({"model": "gpt-fake-1", "input": []}),
            json!({"model": "gpt-fake-1", "messages": []}),
            json!({"model": "gpt-fake-1", "input": [], "messages": []}),
            json!({"model": "gpt-fake-1"}),
        ] {
            let req = request("gpt-fake-1", false);
            assert!(
                matches!(
                    capture_candidate(0, &provider, AdmissionState::Admitted, true, "codex", &req)
                        .binding,
                    BindingFact::Incomparable(IncomparableReason::OperationUndetermined)
                ),
                "body 应封闭为 OperationUndetermined: {body}"
            );
        }
    }

    #[test]
    fn mixed_candidates_close_selection_without_dropping_facts() {
        let req = request("claude-sonnet-4-5-fake", false);
        let claude = claude_provider(
            json!({"env": {"ANTHROPIC_BASE_URL": "https://claude-fake.example.test"}}),
        );
        let codex = codex_provider(json!({"base_url": "https://codex-fake.example.test"}));

        let mut facts = SelectionFacts::new("claude", SelectionMode::FailoverEnabled);
        facts.candidates.push(capture_candidate(
            0,
            &claude,
            AdmissionState::Admitted,
            true,
            "claude",
            &req,
        ));
        facts.candidates.push(capture_candidate(
            1,
            &codex,
            AdmissionState::Admitted,
            true,
            "codex",
            &req,
        ));
        facts
            .candidates
            .push(capture_missing_candidate(2, "fake-missing"));
        facts.mark_capacity();

        // 候选不因绑定失败/缺失而删除；任一不可比 → 整次封闭并记录有序序号
        assert_eq!(facts.candidates.len(), 3);
        assert!(matches!(
            facts.candidates[0].binding,
            BindingFact::Resolved(_)
        ));
        assert_eq!(
            facts.selection_incomparable,
            Some(IncomparableReason::CandidateBindingsIncomparable {
                ordinals: vec![1, 2]
            })
        );

        // 缺失候选占位：未读取熔断器，绑定封闭为 ProviderNotFound
        assert!(!facts.candidates[2].breaker_checked);
        assert!(matches!(
            facts.candidates[2].binding,
            BindingFact::Incomparable(IncomparableReason::ProviderNotFound)
        ));
    }
}
