use crate::{
    app::{
        constant::{
            AUTHORIZATION_BEARER_PREFIX, FINISH_REASON_STOP, OBJECT_CHAT_COMPLETION,
            OBJECT_CHAT_COMPLETION_CHUNK, header_value_chunked, header_value_event_stream,
            header_value_json, header_value_keep_alive, header_value_no_cache_revalidate,
        },
        lazy::{
            AUTH_TOKEN, GENERAL_TIMEZONE, IS_NO_REQUEST_LOGS, IS_UNLIMITED_REQUEST_LOGS,
            KEY_PREFIX, KEY_PREFIX_LEN, REQUEST_LOGS_LIMIT, cursor_api2_chat_url,
            cursor_api2_chat_web_url,
        },
        model::{
            AppConfig, AppState, Chain, ErrorInfo, LogStatus, OptionUsage, Prompt, RequestLog,
            TimingInfo, TokenInfo, UsageCheck, proxy_pool::ProxyPool,
        },
    },
    common::{
        client::{AiServiceRequest, build_request},
        model::{
            ApiStatus, ErrorResponse, error::ChatError, tri::TriState, userinfo::MembershipType,
        },
        utils::{
            TrimNewlines as _, format_time_ms, from_base64, generate_hash, get_available_models,
            get_token_profile, get_token_usage, tokeninfo_to_token, validate_token_and_checksum,
        },
    },
    core::{
        config::KeyConfig,
        constant::Models,
        error::StreamError,
        model::{ChatResponse, Choice, Delta, Message, MessageContent, ModelsResponse, Role},
        stream::decoder::{StreamDecoder, StreamMessage},
    },
    leak::intern_string,
};
use axum::{
    Json,
    body::Body,
    extract::State,
    http::{
        HeaderMap, StatusCode,
        header::{
            AUTHORIZATION, CACHE_CONTROL, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE,
            TRANSFER_ENCODING,
        },
    },
    response::Response,
};
use bytes::Bytes;
use futures::StreamExt;
use prost::Message as _;
use std::{
    borrow::Cow,
    sync::atomic::{AtomicUsize, Ordering},
};
use std::{
    convert::Infallible,
    sync::{Arc, atomic::AtomicBool},
};
use tokio::sync::Mutex;

use super::{constant::FREE_MODELS, model::ChatRequest};

static CURRENT_KEY_INDEX: AtomicUsize = AtomicUsize::new(0);

pub async fn handle_models(
    State(state): State<Arc<Mutex<AppState>>>,
    headers: HeaderMap,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    // 如果没有认证头，返回默认可用模型
    let auth_token = match headers.get(AUTHORIZATION) {
        None => return Ok(Json(ModelsResponse::with_default_models())),
        Some(h) => h
            .to_str()
            .ok()
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or((
                StatusCode::UNAUTHORIZED,
                Json(ChatError::Unauthorized.to_json()),
            ))?,
    };

    let mut is_pri = false;

    // 获取token信息
    let (token, checksum, client_key, client, timezone) = match auth_token {
        // 管理员Token
        token
            if token == AUTH_TOKEN.as_str()
                || (AppConfig::is_share() && AppConfig::share_token_eq(token)) =>
        {
            let state_guard = state.lock().await;
            let token_infos: Vec<_> = state_guard
                .token_manager
                .tokens
                .iter()
                .filter(|t| t.is_enabled())
                .collect();

            if token_infos.is_empty() {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ChatError::NoTokens.to_json()),
                ));
            }

            let index = CURRENT_KEY_INDEX.load(Ordering::Acquire) % token_infos.len();
            let token_info = &token_infos[index];
            is_pri = true;
            (
                token_info.token.clone(),
                token_info.checksum.clone(),
                token_info.client_key.clone(),
                token_info.get_client(),
                token_info.timezone_name(),
            )
        }

        // 动态密钥
        token if AppConfig::get_dynamic_key() && token.starts_with(&*KEY_PREFIX) => {
            from_base64(&token[*KEY_PREFIX_LEN..])
                .and_then(|decoded_bytes| KeyConfig::decode(&decoded_bytes[..]).ok())
                .and_then(|key_config| key_config.auth_token)
                .and_then(tokeninfo_to_token)
                .map(|(token, checksum, client)| {
                    (token, checksum, None, client, GENERAL_TIMEZONE.name())
                })
                .ok_or((
                    StatusCode::UNAUTHORIZED,
                    Json(ChatError::Unauthorized.to_json()),
                ))?
        }

        // 普通用户Token
        token => {
            let (token, checksum) = validate_token_and_checksum(token).ok_or((
                StatusCode::UNAUTHORIZED,
                Json(ChatError::Unauthorized.to_json()),
            ))?;
            (
                token,
                checksum,
                None,
                ProxyPool::get_general_client(),
                GENERAL_TIMEZONE.name(),
            )
        }
    };
    let client_key = client_key.unwrap_or_else(generate_hash);

    // 获取可用模型列表
    let models = get_available_models(client, &token, &checksum, &client_key, timezone, is_pri)
        .await
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                status: ApiStatus::Failure,
                code: Some(StatusCode::INTERNAL_SERVER_ERROR.as_u16()),
                error: Some(Cow::Borrowed("Failed to fetch available models")),
                message: Some(Cow::Borrowed("Unable to get available models")),
            }),
        ))?;

    // 更新模型列表
    Models::update(models).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                status: ApiStatus::Failure,
                code: Some(StatusCode::INTERNAL_SERVER_ERROR.as_u16()),
                error: Some(Cow::Borrowed("Failed to update models")),
                message: Some(Cow::Borrowed(e)),
            }),
        )
    })?;

    Ok(Json(ModelsResponse::new(Models::to_arc())))
}

// 聊天处理函数的签名
pub async fn handle_chat(
    State(state): State<Arc<Mutex<AppState>>>,
    headers: HeaderMap,
    Json(request): Json<ChatRequest>,
) -> Result<Response<Body>, (StatusCode, Json<ErrorResponse>)> {
    let is_search = request.model.ends_with("-online");

    // 验证模型是否支持并获取模型信息
    let model = {
        let model_name = if is_search {
            &request.model[..request.model.len() - 7]
        } else {
            &request.model
        };

        if let Some(model) = Models::find_id(model_name) {
            model
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ChatError::ModelNotSupported(request.model).to_json()),
            ));
        }
    };

    let request_time = chrono::Local::now();

    // 验证请求
    if request.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ChatError::EmptyMessages.to_json()),
        ));
    }

    // 获取并处理认证令牌
    let auth_header = headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix(AUTHORIZATION_BEARER_PREFIX))
        .ok_or((
            StatusCode::UNAUTHORIZED,
            Json(ChatError::Unauthorized.to_json()),
        ))?;

    let mut current_config = KeyConfig::new_with_global();
    let mut is_pri = false;

    // 验证认证token并获取token信息
    let mut now_with_tz = None;
    let (auth_token, checksum, client_key, client, timezone) = match auth_header {
        // 管理员Token验证逻辑
        token
            if token == AUTH_TOKEN.as_str()
                || (AppConfig::is_share() && AppConfig::share_token_eq(token)) =>
        {
            let state_guard = state.lock().await;
            let token_infos: Vec<_> = state_guard
                .token_manager
                .tokens
                .iter()
                .filter(|t| t.is_enabled())
                .collect();

            // 检查是否存在可用的token
            if token_infos.is_empty() {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(ChatError::NoTokens.to_json()),
                ));
            }

            // 轮询选择token
            let index = CURRENT_KEY_INDEX.fetch_add(1, Ordering::SeqCst) % token_infos.len();
            let token_info = &token_infos[index];
            is_pri = true;
            now_with_tz = Some(token_info.now());
            (
                token_info.token.clone(),
                token_info.checksum.clone(),
                token_info.client_key.clone(),
                token_info.get_client(),
                token_info.timezone_name(),
            )
        }

        token if AppConfig::get_dynamic_key() && token.starts_with(&*KEY_PREFIX) => {
            from_base64(&token[*KEY_PREFIX_LEN..])
                .and_then(|decoded_bytes| KeyConfig::decode(&decoded_bytes[..]).ok())
                .and_then(|key_config| {
                    key_config.copy_without_auth_token(&mut current_config);
                    key_config.auth_token
                })
                .and_then(|info| {
                    tokeninfo_to_token(info)
                        .map(|(e1, e2, e3)| (e1, e2, None, e3, GENERAL_TIMEZONE.name()))
                })
                .ok_or((
                    StatusCode::UNAUTHORIZED,
                    Json(ChatError::Unauthorized.to_json()),
                ))?
        }

        // 普通用户Token验证逻辑
        token => {
            let (token, checksum) = validate_token_and_checksum(token).ok_or((
                StatusCode::UNAUTHORIZED,
                Json(ChatError::Unauthorized.to_json()),
            ))?;
            (
                token,
                checksum,
                None,
                ProxyPool::get_general_client(),
                GENERAL_TIMEZONE.name(),
            )
        }
    };
    let client_key = client_key.unwrap_or_else(generate_hash);

    let current_config = current_config;

    let current_id: u64;

    // 更新请求日志
    if !*IS_NO_REQUEST_LOGS {
        let state_clone = state.clone();
        let mut state = state.lock().await;
        state.request_manager.total_requests += 1;
        state.request_manager.active_requests += 1;

        let mut need_profile_check = false;

        for log in state.request_manager.request_logs.iter().rev() {
            if log.token_info.token == auth_token {
                if let Some(profile) = &log.token_info.profile {
                    if profile.stripe.membership_type == MembershipType::Free {
                        need_profile_check = if FREE_MODELS.contains(&model.id) {
                            profile
                                .usage
                                .standard
                                .max_requests
                                .is_some_and(|max| profile.usage.standard.num_requests >= max)
                        } else {
                            profile
                                .usage
                                .premium
                                .max_requests
                                .is_some_and(|max| profile.usage.premium.num_requests >= max)
                        };
                    }
                    break;
                }
            }
        }

        // 处理检查结果
        if need_profile_check {
            state.request_manager.active_requests -= 1;
            state.request_manager.error_requests += 1;
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(ChatError::Unauthorized.to_json()),
            ));
        }

        let next_id = state
            .request_manager
            .request_logs
            .back()
            .map_or(1, |log| log.id + 1);
        current_id = next_id;

        // 如果需要获取用户使用情况,创建后台任务获取profile
        if model.is_usage_check(UsageCheck::from_proto(
            current_config.usage_check_models.as_ref(),
        )) {
            let auth_token_clone = auth_token.clone();
            let state_clone = state_clone.clone();
            let log_id = next_id;
            let client = client.clone();

            tokio::spawn(async move {
                let profile = get_token_profile(client, &auth_token_clone, is_pri).await;
                let mut state = state_clone.lock().await;

                // 先找到所有需要更新的位置的索引
                let token_info_idx = state
                    .token_manager
                    .tokens
                    .iter()
                    .position(|info| info.token == auth_token_clone);

                let log_idx = state
                    .request_manager
                    .request_logs
                    .iter()
                    .rposition(|log| log.id == log_id);

                // 根据索引更新
                match (token_info_idx, log_idx) {
                    (Some(t_idx), Some(l_idx)) => {
                        state.token_manager.tokens[t_idx].profile = profile.clone();
                        state.request_manager.request_logs[l_idx].token_info.profile = profile;
                    }
                    (Some(t_idx), None) => {
                        state.token_manager.tokens[t_idx].profile = profile;
                    }
                    (None, Some(l_idx)) => {
                        state.request_manager.request_logs[l_idx].token_info.profile = profile;
                    }
                    (None, None) => {}
                }
            });
        }

        state.request_manager.request_logs.push_back(RequestLog {
            id: next_id,
            timestamp: request_time,
            model: intern_string(request.model),
            token_info: TokenInfo {
                token: auth_token.clone(),
                checksum: checksum.clone(),
                status: Default::default(),
                client_key: None,
                profile: None,
                tags: None,
            },
            chain: None,
            timing: TimingInfo { total: 0.0 },
            stream: request.stream,
            status: LogStatus::Pending,
            error: ErrorInfo::None,
        });

        if !*IS_UNLIMITED_REQUEST_LOGS
            && state.request_manager.request_logs.len() > *REQUEST_LOGS_LIMIT
        {
            state.request_manager.request_logs.remove(0);
        }
    } else {
        current_id = 0;
    }

    // 将消息转换为hex格式
    let hex_data = match super::adapter::encode_chat_message(
        request.messages,
        now_with_tz,
        model,
        current_config.disable_vision(),
        current_config.enable_slow_pool(),
        is_search,
    )
    .await
    {
        Ok(data) => data,
        Err(e) => {
            let mut state = state.lock().await;
            if let Some(log) = state
                .request_manager
                .request_logs
                .iter_mut()
                .rev()
                .find(|log| log.id == current_id)
            {
                log.status = LogStatus::Failure;
                log.error = ErrorInfo::Error(intern_string(e.to_string()));
            }
            state.request_manager.active_requests -= 1;
            state.request_manager.error_requests += 1;
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    ChatError::RequestFailed("Failed to encode chat message".to_string()).to_json(),
                ),
            ));
        }
    };

    // 构建请求客户端
    let trace_id = uuid::Uuid::new_v4();
    let req = build_request(AiServiceRequest {
        client: client.clone(),
        auth_token: auth_token.as_str(),
        checksum: checksum.as_str(),
        client_key: client_key.as_str(),
        url: if is_search {
            cursor_api2_chat_web_url(is_pri)
        } else {
            cursor_api2_chat_url(is_pri)
        },
        is_stream: true,
        timezone,
        trace_id: &trace_id.to_string(),
        is_pri,
    });
    let trace_id = trace_id.simple();
    // 发送请求
    let response = req.body(hex_data).send().await;

    // 处理请求结果
    let response = match response {
        Ok(resp) => {
            // 更新请求日志为成功
            {
                let mut state = state.lock().await;
                if let Some(log) = state
                    .request_manager
                    .request_logs
                    .iter_mut()
                    .rev()
                    .find(|log| log.id == current_id)
                {
                    log.status = LogStatus::Success;
                }
            }
            resp
        }
        Err(mut e) => {
            e = e.without_url();
            // 更新请求日志为失败
            {
                let mut state = state.lock().await;
                if let Some(log) = state
                    .request_manager
                    .request_logs
                    .iter_mut()
                    .rev()
                    .find(|log| log.id == current_id)
                {
                    log.status = LogStatus::Failure;
                    log.error = ErrorInfo::Error(intern_string(e.to_string()));
                }
                state.request_manager.active_requests -= 1;
                state.request_manager.error_requests += 1;
            }

            // 根据错误类型返回不同的状态码
            let status_code = if e.is_timeout() {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };

            return Err((
                status_code,
                Json(ChatError::RequestFailed(e.to_string()).to_json()),
            ));
        }
    };

    // 释放活动请求计数
    {
        let mut state = state.lock().await;
        state.request_manager.active_requests -= 1;
    }

    let convert_web_ref = current_config.include_web_references();

    if request.stream {
        let response_id = format!("chatcmpl-{trace_id}");
        let is_start = Arc::new(AtomicBool::new(true));
        let start_time = std::time::Instant::now();
        let decoder = Arc::new(Mutex::new(StreamDecoder::new()));
        let need_usage = Arc::new(Mutex::new(NeedUsage::Need {
            is_need: request.stream_options.is_some_and(|opt| opt.include_usage),
            client,
            auth_token,
            checksum,
            client_key,
            timezone,
            is_pri,
        }));
        let usage_uuid = Arc::new(Mutex::new(None));

        // 定义消息处理器的上下文结构体
        struct MessageProcessContext<'a> {
            response_id: &'a str,
            model: &'static str,
            is_start: &'a AtomicBool,
            start_time: std::time::Instant,
            state: &'a Mutex<AppState>,
            current_id: u64,
            usage_uuid: &'a Mutex<Option<String>>,
            need_usage: &'a Mutex<NeedUsage>,
            created: i64,
        }

        #[derive(Default)]
        enum NeedUsage {
            #[default]
            Taked,
            Need {
                is_need: bool,
                client: reqwest::Client,
                auth_token: String,
                checksum: String,
                client_key: String,
                timezone: &'static str,
                is_pri: bool,
            },
        }

        impl NeedUsage {
            #[inline(always)]
            const fn is_need(&self) -> bool {
                match self {
                    Self::Taked => false,
                    Self::Need { is_need, .. } => *is_need,
                }
            }

            #[inline(always)]
            fn take(&mut self) -> Self {
                std::mem::take(self)
            }
        }

        // 处理消息并生成响应数据的辅助函数
        async fn process_messages(
            messages: Vec<StreamMessage>,
            ctx: &MessageProcessContext<'_>,
        ) -> Vec<u8> {
            let mut response_data = Vec::new();

            for message in messages {
                match message {
                    StreamMessage::Content(text) => {
                        let is_first = ctx.is_start.load(Ordering::Acquire);

                        let response = ChatResponse {
                            id: ctx.response_id,
                            object: OBJECT_CHAT_COMPLETION_CHUNK,
                            created: chrono::Utc::now().timestamp(),
                            model: if is_first { Some(ctx.model) } else { None },
                            choices: vec![Choice {
                                index: 0,
                                message: None,
                                delta: Some(Delta {
                                    role: if is_first {
                                        Some(Role::Assistant)
                                    } else {
                                        None
                                    },
                                    content: if is_first {
                                        ctx.is_start.store(false, Ordering::Release);
                                        Some(text.trim_leading_newlines())
                                    } else {
                                        Some(text)
                                    },
                                }),
                                logprobs: None,
                                finish_reason: None,
                            }],
                            usage: if ctx.need_usage.lock().await.is_need() {
                                TriState::Null
                            } else {
                                TriState::None
                            },
                        };

                        response_data.extend_from_slice(b"data: ");
                        response_data.extend_from_slice(&serde_json::to_vec(&response).unwrap());
                        response_data.extend_from_slice(b"\n\n");
                    }
                    StreamMessage::Usage(usage_uuid) => {
                        if !usage_uuid.is_empty() && ctx.need_usage.lock().await.is_need() {
                            *ctx.usage_uuid.lock().await = Some(usage_uuid);
                        }
                    }
                    StreamMessage::StreamEnd => {
                        // 计算总时间和首次片段时间
                        let total_time = ctx.start_time.elapsed().as_secs_f64();

                        {
                            let mut state = ctx.state.lock().await;
                            if let Some(log) = state
                                .request_manager
                                .request_logs
                                .iter_mut()
                                .rev()
                                .find(|log| log.id == ctx.current_id)
                            {
                                log.timing.total = format_time_ms(total_time);
                            }
                        }

                        let response = ChatResponse {
                            id: ctx.response_id,
                            object: OBJECT_CHAT_COMPLETION_CHUNK,
                            created: chrono::Utc::now().timestamp(),
                            model: None,
                            choices: vec![Choice {
                                index: 0,
                                message: None,
                                delta: Some(Delta {
                                    role: None,
                                    content: None,
                                }),
                                logprobs: None,
                                finish_reason: Some(FINISH_REASON_STOP.to_string()),
                            }],
                            usage: if ctx.need_usage.lock().await.is_need() {
                                TriState::Null
                            } else {
                                TriState::None
                            },
                        };
                        response_data.extend_from_slice(b"data: ");
                        response_data.extend_from_slice(&serde_json::to_vec(&response).unwrap());
                        response_data.extend_from_slice(b"\n\n");
                        if let Some(usage_uuid) = ctx.usage_uuid.lock().await.take() {
                            if let NeedUsage::Need {
                                is_need,
                                client,
                                auth_token,
                                checksum,
                                client_key,
                                timezone,
                                is_pri,
                            } = ctx.need_usage.lock().await.take()
                            {
                                let usage = if *crate::app::lazy::REAL_USAGE {
                                    let usage = tokio::spawn(get_token_usage(
                                        client, auth_token, checksum, client_key, timezone, is_pri,
                                        usage_uuid,
                                    ))
                                    .await
                                    .unwrap_or_default();
                                    if let Some(ref usage) = usage {
                                        let mut state = ctx.state.lock().await;
                                        if let Some(log) = state
                                            .request_manager
                                            .request_logs
                                            .iter_mut()
                                            .rev()
                                            .find(|log| log.id == ctx.current_id)
                                        {
                                            if let Some(chain) = &mut log.chain {
                                                chain.usage = OptionUsage::Uasge {
                                                    input: usage.prompt_tokens,
                                                    output: usage.completion_tokens,
                                                }
                                            }
                                        }
                                    }
                                    usage
                                } else {
                                    None
                                };

                                if is_need {
                                    let response = ChatResponse {
                                        id: ctx.response_id,
                                        object: OBJECT_CHAT_COMPLETION_CHUNK,
                                        created: ctx.created,
                                        model: None,
                                        choices: vec![],
                                        usage: TriState::Some(usage.unwrap_or_default()),
                                    };
                                    response_data.extend_from_slice(b"data: ");
                                    response_data
                                        .extend_from_slice(&serde_json::to_vec(&response).unwrap());
                                    response_data.extend_from_slice(b"\n\n");
                                }
                            };
                        }
                    }
                    StreamMessage::Debug(debug_prompt) => {
                        if let Ok(mut state) = ctx.state.try_lock() {
                            if let Some(log) = state
                                .request_manager
                                .request_logs
                                .iter_mut()
                                .rev()
                                .find(|log| log.id == ctx.current_id)
                            {
                                if log.chain.is_some() {
                                    crate::debug_println!("UB!1 {debug_prompt:?}");
                                    // chain.prompt.push_str(&debug_prompt);
                                } else {
                                    log.chain = Some(Chain {
                                        prompt: Prompt::new(debug_prompt),
                                        delays: None,
                                        usage: OptionUsage::None,
                                    });
                                }
                            }
                        }
                    }
                    _ => {} // 忽略其他消息类型
                }
            }

            response_data
        }

        // 首先处理stream直到获得第一个结果
        let mut stream = response.bytes_stream();
        while !decoder.lock().await.is_first_result_ready() {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    if let Err(StreamError::ChatError(error)) =
                        decoder.lock().await.decode(&chunk, convert_web_ref)
                    {
                        let error_response = error.into_error_response();
                        // 更新请求日志为失败
                        {
                            let mut state = state.lock().await;
                            if let Some(log) = state
                                .request_manager
                                .request_logs
                                .iter_mut()
                                .rev()
                                .find(|log| log.id == current_id)
                            {
                                log.status = LogStatus::Failure;
                                log.error =
                                    ErrorInfo::Error(intern_string(error_response.native_code()));
                                if let Some(detail) = error_response.details() {
                                    log.error.add_detail(&detail)
                                }
                                log.timing.total =
                                    format_time_ms(start_time.elapsed().as_secs_f64());
                                state.request_manager.error_requests += 1;
                            }
                        }
                        return Err((
                            error_response.status_code(),
                            Json(error_response.into_common()),
                        ));
                    }
                }
                Some(Err(e)) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            ChatError::RequestFailed(format!("Failed to read response chunk: {e}"))
                                .to_json(),
                        ),
                    ));
                }
                None => {
                    // 更新请求日志为失败
                    {
                        let mut state = state.lock().await;
                        if let Some(log) = state
                            .request_manager
                            .request_logs
                            .iter_mut()
                            .rev()
                            .find(|log| log.id == current_id)
                        {
                            log.status = LogStatus::Failure;
                            log.error = ErrorInfo::Error(intern_string("Empty stream response"));
                            state.request_manager.error_requests += 1;
                        }
                    }
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(
                            ChatError::RequestFailed("Empty stream response".to_string()).to_json(),
                        ),
                    ));
                }
            }
        }

        let created = Arc::new(std::sync::OnceLock::new());

        // 处理后续的stream
        let stream = stream
            .then({
                let decoder = decoder.clone();
                let state = state.clone();

                move |chunk| {
                    let decoder = decoder.clone();
                    let response_id = response_id.clone();
                    let is_start = is_start.clone();
                    let state = state.clone();
                    let need_usage = need_usage.clone();
                    let usage_uuid = usage_uuid.clone();
                    let created = created.clone();

                    async move {
                        let chunk = match chunk {
                            Ok(c) => c,
                            Err(_) => {
                                // crate::debug_println!("Find chunk error: {e}");
                                return Ok::<_, Infallible>(Bytes::new());
                            }
                        };

                        let ctx = MessageProcessContext {
                            response_id: &response_id,
                            model: model.id,
                            is_start: &is_start,
                            start_time,
                            state: &state,
                            current_id,
                            usage_uuid: &usage_uuid,
                            need_usage: &need_usage,
                            created: *created.get_or_init(|| chrono::Utc::now().timestamp()),
                        };

                        // 使用decoder处理chunk
                        let messages = match decoder.lock().await.decode(&chunk, convert_web_ref) {
                            Ok(msgs) => msgs,
                            Err(e) => {
                                match e {
                                    // 处理普通空流错误
                                    StreamError::EmptyStream => {
                                        eprintln!(
                                            "[警告] Stream error: empty stream (连续计数: {})",
                                            decoder.lock().await.get_empty_stream_count()
                                        );
                                        return Ok(Bytes::new());
                                    }
                                    StreamError::ChatError(e) => {
                                        return Ok(Bytes::from(
                                            serde_json::to_string(
                                                &e.into_error_response().into_common(),
                                            )
                                            .unwrap(),
                                        ));
                                    }
                                    // 处理其他错误
                                    _ => {
                                        eprintln!("[警告] Stream error: {e}");
                                        return Ok(Bytes::new());
                                    }
                                }
                            }
                        };

                        let mut response_data = Vec::new();

                        if let Some(first_msg) = decoder.lock().await.take_first_result() {
                            let first_response = process_messages(first_msg, &ctx).await;
                            response_data.extend_from_slice(&first_response);
                        }

                        let current_response = process_messages(messages, &ctx).await;
                        if !current_response.is_empty() {
                            response_data.extend_from_slice(&current_response);
                        }

                        Ok(Bytes::from(response_data))
                    }
                }
            })
            .chain(futures::stream::once(async move {
                // 更新delays
                let mut state = state.lock().await;
                if let Some(log) = state
                    .request_manager
                    .request_logs
                    .iter_mut()
                    .rev()
                    .find(|log| log.id == current_id)
                {
                    if let Some(chain) = &mut log.chain {
                        chain.delays = decoder.lock().await.take_content_delays();
                    } else {
                        log.chain = Some(Chain {
                            prompt: Prompt::Origin(String::new()),
                            delays: decoder.lock().await.take_content_delays(),
                            usage: OptionUsage::None,
                        });
                    }
                }
                Ok(Bytes::from_static(b"data: [DONE]\n\n"))
            }));

        Ok(Response::builder()
            .header(CACHE_CONTROL, header_value_no_cache_revalidate())
            .header(CONNECTION, header_value_keep_alive())
            .header(CONTENT_TYPE, header_value_event_stream())
            .header(TRANSFER_ENCODING, header_value_chunked())
            .body(Body::from_stream(stream))
            .unwrap())
    } else {
        // 非流式响应
        let start_time = std::time::Instant::now();
        let mut decoder = StreamDecoder::new().no_first_cache();
        let mut full_text = String::with_capacity(1024);
        let mut stream = response.bytes_stream();
        let mut prompt = Prompt::None;
        let mut usage_uuid = String::new();

        // 逐个处理chunks
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        ChatError::RequestFailed(format!("Failed to read response chunk: {e}"))
                            .to_json(),
                    ),
                )
            })?;

            // 立即处理当前chunk
            match decoder.decode(&chunk, convert_web_ref) {
                Ok(messages) => {
                    for message in messages {
                        match message {
                            StreamMessage::Content(text) => {
                                full_text.push_str(&text);
                            }
                            StreamMessage::Usage(uuid) => {
                                usage_uuid = uuid;
                            }
                            StreamMessage::Debug(debug_prompt) => {
                                if prompt.is_none() {
                                    prompt = Prompt::new(debug_prompt);
                                } else {
                                    crate::debug_println!("UB!2 {debug_prompt:?}");
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Err(StreamError::ChatError(error)) => {
                    let error_response = error.into_error_response();
                    {
                        let mut state = state.lock().await;
                        if let Some(log) = state
                            .request_manager
                            .request_logs
                            .iter_mut()
                            .rev()
                            .find(|log| log.id == current_id)
                        {
                            log.status = LogStatus::Failure;
                            log.error =
                                ErrorInfo::Error(intern_string(error_response.native_code()));
                            if let Some(detail) = error_response.details() {
                                log.error.add_detail(&detail)
                            }
                            state.request_manager.error_requests += 1;
                        }
                    }
                    return Err((
                        error_response.status_code(),
                        Json(error_response.into_common()),
                    ));
                }
                Err(e) => {
                    {
                        let mut state = state.lock().await;
                        if let Some(log) = state
                            .request_manager
                            .request_logs
                            .iter_mut()
                            .rev()
                            .find(|log| log.id == current_id)
                        {
                            log.status = LogStatus::Failure;
                            log.error = ErrorInfo::Error(intern_string(e.to_string()));
                            state.request_manager.error_requests += 1;
                        }
                    }
                    let error_response = ErrorResponse {
                        status: ApiStatus::Error,
                        code: Some(500),
                        error: Some(Cow::Owned(e.to_string())),
                        message: None,
                    };
                    return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)));
                }
            }
        }

        // 检查响应是否为空
        if full_text.is_empty() {
            // 更新请求日志为失败
            {
                let mut state = state.lock().await;
                if let Some(log) = state
                    .request_manager
                    .request_logs
                    .iter_mut()
                    .rev()
                    .find(|log| log.id == current_id)
                {
                    log.status = LogStatus::Failure;
                    log.error = ErrorInfo::Error(intern_string("Empty response received"));
                    state.request_manager.error_requests += 1;
                }
            }
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ChatError::RequestFailed("Empty response received".to_string()).to_json()),
            ));
        }

        let (usage1, usage2) = if !usage_uuid.is_empty() {
            let result = get_token_usage(
                client, auth_token, checksum, client_key, timezone, is_pri, usage_uuid,
            )
            .await;
            let result2 = match result {
                Some(ref usage) => OptionUsage::Uasge {
                    input: usage.prompt_tokens,
                    output: usage.completion_tokens,
                },
                None => OptionUsage::None,
            };
            (result, result2)
        } else {
            (None, OptionUsage::None)
        };

        let response_data = ChatResponse {
            id: &format!("chatcmpl-{trace_id}"),
            object: OBJECT_CHAT_COMPLETION,
            created: chrono::Utc::now().timestamp(),
            model: Some(model.id),
            choices: vec![Choice {
                index: 0,
                message: Some(Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(full_text.trim_leading_newlines()),
                }),
                delta: None,
                logprobs: None,
                finish_reason: Some(FINISH_REASON_STOP.to_string()),
            }],
            usage: TriState::Some(usage1.unwrap_or_default()),
        };

        {
            // 更新请求日志时间信息和状态
            let total_time = format_time_ms(start_time.elapsed().as_secs_f64());
            let mut state = state.lock().await;
            if let Some(log) = state
                .request_manager
                .request_logs
                .iter_mut()
                .rev()
                .find(|log| log.id == current_id)
            {
                log.timing.total = total_time;
                log.status = LogStatus::Success;
                log.chain = Some(Chain {
                    prompt,
                    delays: decoder.take_content_delays(),
                    usage: usage2,
                });
            }
        }

        let data = serde_json::to_vec(&response_data).unwrap();
        Ok(Response::builder()
            .header(CACHE_CONTROL, header_value_no_cache_revalidate())
            .header(CONNECTION, header_value_keep_alive())
            .header(CONTENT_TYPE, header_value_json())
            .header(CONTENT_LENGTH, data.len())
            .body(Body::from(data))
            .unwrap())
    }
}

// pub async fn handle_completion(
//     State(state): State<Arc<Mutex<AppState>>>,
//     headers: HeaderMap,
//     Json(request): Json<ChatRequest>,
// ) -> Result<Response<Body>, (StatusCode, Json<ErrorResponse>)> {

// }
