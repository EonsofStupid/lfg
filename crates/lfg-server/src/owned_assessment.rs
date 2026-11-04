//! Opt-in local inference through the owned byte ONNX runtime. Private prompts and
//! generated text are handled in memory only and are never logged here.
use std::{
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{ConnectInfo, DefaultBodyLimit, State},
    http::{HeaderMap, HeaderValue, StatusCode, header::CACHE_CONTROL},
    middleware::map_response,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
    sync::{OwnedSemaphorePermit, Semaphore},
};

const CONFIG_ENV: &str = "LFG_OWNED_ASSESSMENT_CONFIG";
const TOKEN_ENV: &str = "LFG_OWNED_ASSESSMENT_TOKEN";
const MAX_CONFIG_BYTES: u64 = 128 * 1024;

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Backend {
    LegacyFullPrefix,
    CachedOnnx,
}

fn default_backend() -> Backend {
    Backend::LegacyFullPrefix
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ArtifactMode {
    Directory,
    Embedded,
}

fn default_artifact_mode() -> ArtifactMode {
    ArtifactMode::Directory
}

#[derive(Clone)]
struct Api {
    listener: SocketAddr,
    token: Option<Arc<String>>,
    allowed_hosts: Arc<Vec<String>>,
    allowed_origins: Arc<Vec<String>>,
    service: Option<Arc<Service>>,
    unavailable_reason: Option<&'static str>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    schema_version: String,
    enabled: bool,
    #[serde(default = "default_backend")]
    backend: Backend,
    #[serde(default = "default_artifact_mode")]
    artifact_mode: ArtifactMode,
    #[serde(default)]
    artifact_directory: Option<PathBuf>,
    #[serde(default)]
    expected_cached_manifest_sha256: Option<String>,
    #[serde(default)]
    max_cached_graph_bytes: Option<u64>,
    #[serde(default, rename = "maxCacheStateBytesPerRequest")]
    max_cache_bytes: Option<u64>,
    framing_file: PathBuf,
    training_manifest_file: PathBuf,
    runtime_config_file: PathBuf,
    max_runtime_config_bytes: u64,
    expected_model_sha256: String,
    expected_framing_sha256: String,
    max_metadata_bytes: u64,
    required_output_fields: Vec<RequiredField>,
    max_http_body_bytes: usize,
    max_request_bytes: usize,
    max_context_bytes: usize,
    max_generated_bytes: usize,
    max_worker_output_bytes: usize,
    timeout_ms: u64,
    max_concurrent: usize,
    worker_threads: usize,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RequiredField {
    name: String,
    value_type: OutputType,
    #[serde(default)]
    non_empty: bool,
    min_integer: Option<i64>,
    max_integer: Option<i64>,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum OutputType {
    String,
    Number,
    Integer,
    Boolean,
    Array,
    Object,
}

#[derive(Clone)]
struct Service {
    config: Config,
    backend: Backend,
    engine: Value,
    system: String,
    framing: Framing,
    runtime_line_bytes: usize,
    runtime_sha256: String,
    runtime_max_bytes: u64,
    permits: Arc<Semaphore>,
}

#[derive(Clone, Deserialize)]
struct Framing {
    system_prefix: String,
    system_suffix: String,
    user_prefix: String,
    user_suffix: String,
    assistant_prefix: String,
    assistant_suffix: String,
}

#[derive(Deserialize)]
struct FramingEnvelope {
    system: String,
    framing: Framing,
}

#[derive(Deserialize)]
struct RuntimeConfig {
    schema_version: String,
    max_manifest_bytes: u64,
    max_model_bytes: u64,
    max_context: usize,
    max_line_bytes: usize,
    max_generation_bytes: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InferRequest {
    text: String,
    #[serde(default)]
    context: Option<String>,
    #[serde(default)]
    context_sha256: Option<String>,
    #[serde(flatten)]
    _extensions: serde_json::Map<String, Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceResponse {
    schema_version: &'static str,
    status: &'static str,
    engine: Option<Value>,
    elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<Value>,
    measurement: OwnedRunMeasurement,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OwnedRunMeasurement {
    schema_version: &'static str,
    status: &'static str,
    engine_id: Option<String>,
    stop_reason: Option<String>,
    handler_elapsed_ms: u64,
    output_text_utf8_bytes: Option<usize>,
    worker_started: Option<bool>,
    worker_completed_successfully: Option<bool>,
}

pub(crate) fn routes(listener: SocketAddr) -> Router {
    let api = Api::new(listener);
    owned_routes(api)
}

fn owned_routes(api: Api) -> Router {
    let max_body = api
        .service
        .as_ref()
        .map_or(4096, |service| service.config.max_http_body_bytes);
    Router::new()
        .route("/api/lfg/assess/owned/status", get(status))
        .route("/api/lfg/assess/owned", post(infer))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(map_response(|mut response: Response| async move {
            response
                .headers_mut()
                .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
            response
        }))
        .with_state(api)
}

impl Api {
    fn new(listener: SocketAddr) -> Self {
        let allowed_hosts = list_env("LFG_OWNED_ASSESSMENT_ALLOWED_HOSTS", true);
        let allowed_origins = list_env("LFG_OWNED_ASSESSMENT_ALLOWED_ORIGINS", false);
        let token = std::env::var(TOKEN_ENV)
            .ok()
            .filter(|value| super::local_access::valid_token(value))
            .map(Arc::new);
        let (service, unavailable_reason) = if !listener.ip().is_loopback() {
            (None, Some("loopback_listener_required"))
        } else if token.is_none() {
            (None, Some("session_token_unavailable"))
        } else if !cfg!(feature = "owned-byte-onnx") {
            (None, Some("owned_runtime_not_built"))
        } else {
            match std::env::var_os(CONFIG_ENV) {
                None => (None, Some("service_disabled")),
                Some(path) => match load_service(PathBuf::from(path)) {
                    Ok(Some(service)) => (Some(Arc::new(service)), None),
                    Ok(None) => (None, Some("service_disabled")),
                    Err(reason) => (None, Some(reason)),
                },
            }
        };
        Self {
            listener,
            token,
            allowed_hosts: Arc::new(allowed_hosts),
            allowed_origins: Arc::new(allowed_origins),
            service,
            unavailable_reason,
        }
    }

    fn authorize(&self, peer: SocketAddr, headers: &HeaderMap) -> Result<(), StatusCode> {
        super::local_access::authorize(
            self.listener,
            peer,
            headers,
            self.token.as_deref().map(String::as_str),
            &self.allowed_hosts,
            &self.allowed_origins,
        )
        .map_err(|error| match error {
            super::local_access::LocalAccessError::InvalidToken => StatusCode::FORBIDDEN,
            super::local_access::LocalAccessError::InvalidHost
            | super::local_access::LocalAccessError::InvalidOrigin => StatusCode::FORBIDDEN,
            super::local_access::LocalAccessError::NonLoopback => StatusCode::FORBIDDEN,
            super::local_access::LocalAccessError::TokenUnavailable => {
                StatusCode::SERVICE_UNAVAILABLE
            }
        })
    }
}

async fn status(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(code) = api.authorize(peer, &headers) {
        return (
            code,
            Json(
                json!({"schemaVersion":"lfg.owned-assessment-status.v1","enabled":false,"status":"unavailable","reason":auth_reason(code)}),
            ),
        );
    }
    let status = if api.service.is_some() {
        "ready"
    } else if api.unavailable_reason == Some("service_disabled") {
        "disabled"
    } else {
        "unavailable"
    };
    let service = api.service.as_ref();
    let body = json!({
        "schemaVersion":"lfg.owned-assessment-status.v1",
        "enabled":service.is_some(),
        "status":status,
        "engine":service.map(|s| s.engine.clone()),
        "limits":service.map(|s| json!({
            "backend":match s.backend { Backend::LegacyFullPrefix => "legacy_full_prefix", Backend::CachedOnnx => "cached_onnx" },
            "maxRequestBytes":s.config.max_request_bytes,
            "maxContextBytes":s.config.max_context_bytes,
            "maxGeneratedBytes":s.config.max_generated_bytes,
            "maxCachedGraphBytes":s.config.max_cached_graph_bytes,
            "maxCacheStateBytesPerRequest":s.config.max_cache_bytes,
            "timeoutMs":s.config.timeout_ms,
            "maxConcurrent":s.config.max_concurrent,
            "workerThreads":s.config.worker_threads
        })),
        "reason":if service.is_none() { api.unavailable_reason } else { None }
    });
    (StatusCode::OK, Json(body))
}

async fn infer(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let started = Instant::now();
    if let Err(code) = api.authorize(peer, &headers) {
        return (
            code,
            Json(
                json!({"schemaVersion":"lfg.owned-assessment.v1","status":"unavailable","engine":null,"elapsedMs":started.elapsed().as_millis() as u64,"measurement":{"schemaVersion":"lfg.owned-run-measurement.v1","status":"unavailable","engineId":null,"stopReason":null,"handlerElapsedMs":started.elapsed().as_millis() as u64,"outputTextUtf8Bytes":null,"workerStarted":false,"workerCompletedSuccessfully":false},"reason":auth_reason(code)}),
            ),
        );
    }
    let Some(service) = api.service.as_ref() else {
        return (
            StatusCode::OK,
            Json(
                json!({"schemaVersion":"lfg.owned-assessment.v1","status":"unavailable","engine":null,"elapsedMs":started.elapsed().as_millis() as u64,"measurement":{"schemaVersion":"lfg.owned-run-measurement.v1","status":"unavailable","engineId":null,"stopReason":null,"handlerElapsedMs":started.elapsed().as_millis() as u64,"outputTextUtf8Bytes":null,"workerStarted":false,"workerCompletedSuccessfully":false},"reason":api.unavailable_reason.unwrap_or("unavailable")}),
            ),
        );
    };
    if body.len() > service.config.max_http_body_bytes {
        return response(
            "invalid_input",
            Some(service.engine.clone()),
            started,
            None,
            None,
            None,
        );
    }
    let request: InferRequest = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(_) => {
            return response(
                "invalid_input",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    if request.text.is_empty() || request.text.len() > service.config.max_request_bytes {
        return response(
            "invalid_input",
            Some(service.engine.clone()),
            started,
            None,
            None,
            None,
        );
    }
    let user_text = match render_user_text(
        request.context.as_deref(),
        request.context_sha256.as_deref(),
        &request.text,
        service.config.max_context_bytes,
    ) {
        Ok(value) => value,
        Err(()) => {
            return response(
                "invalid_input",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    // The context bytes have now been checked against this digest. Return the
    // digest only so a caller can explicitly bind its own feedback record.
    let verified_context_sha256 = request.context_sha256.clone();
    let permit: OwnedSemaphorePermit = match service.permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return add_context_digest(
                response(
                    "busy",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                verified_context_sha256.as_deref(),
            );
        }
    };
    let result = run_inference(service.clone(), user_text, started).await;
    drop(permit);
    add_context_digest(result, verified_context_sha256.as_deref())
}

fn render_user_text(
    context: Option<&str>,
    expected_sha256: Option<&str>,
    request: &str,
    max_context_bytes: usize,
) -> Result<String, ()> {
    if request.is_empty() {
        return Err(());
    }
    match (context, expected_sha256) {
        (None, None) => Ok(request.to_owned()),
        (Some(context), Some(expected)) if !context.is_empty() => {
            if context.len() > max_context_bytes || digest(context.as_bytes()) != expected {
                return Err(());
            }
            Ok(format!("{context}\n\n{request}"))
        }
        _ => Err(()),
    }
}

fn model_prompt(system: &str, framing: &Framing, user: &str) -> String {
    format!(
        "{}{}{}{}{}{}{}",
        framing.system_prefix,
        system,
        framing.system_suffix,
        framing.user_prefix,
        user,
        framing.user_suffix,
        framing.assistant_prefix,
    )
}

async fn run_inference(
    service: Arc<Service>,
    text: String,
    started: Instant,
) -> (StatusCode, Json<Value>) {
    let config = &service.config;
    let prompt = model_prompt(&service.system, &service.framing, &text);
    if prompt.len() > config.max_context_bytes {
        return response(
            "context_limit",
            Some(service.engine.clone()),
            started,
            None,
            None,
            None,
        );
    }
    let input = match serde_json::to_vec(&json!({"id":"owned-assessment","text":prompt})) {
        Ok(mut value) => {
            value.push(b'\n');
            value
        }
        Err(_) => {
            return response(
                "inference_error",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    if input.len() > service.runtime_line_bytes {
        return response(
            "invalid_input",
            Some(service.engine.clone()),
            started,
            None,
            None,
            None,
        );
    }
    let exe = match std::env::current_exe() {
        Ok(value) => value,
        Err(_) => {
            return response(
                "unavailable",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    let runtime = config.runtime_config_file.clone();
    let mut command = Command::new(exe);
    match service.backend {
        Backend::LegacyFullPrefix => {
            let Some(artifact) = config.artifact_directory.as_ref() else {
                return response(
                    "unavailable",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                );
            };
            command
                .args(["owned-byte-generate", "--experimental", "--artifact"])
                .arg(artifact)
                .arg("--manifest-sha256")
                .arg(
                    service.engine["modelManifestSha256"]
                        .as_str()
                        .unwrap_or_default(),
                );
        }
        Backend::CachedOnnx => {
            let manifest_sha = service.engine["cachedManifestSha256"]
                .as_str()
                .unwrap_or_default();
            let model_sha = service.engine["modelSha256"].as_str().unwrap_or_default();
            match config.artifact_mode {
                ArtifactMode::Directory => {
                    let Some(artifact) = config.artifact_directory.as_ref() else {
                        return response(
                            "unavailable",
                            Some(service.engine.clone()),
                            started,
                            None,
                            None,
                            None,
                        );
                    };
                    command
                        .args(["owned-byte-cached-generate", "--experimental", "--artifact"])
                        .arg(artifact)
                        .arg("--manifest-sha256")
                        .arg(manifest_sha)
                        .arg("--model-sha256")
                        .arg(model_sha)
                        .arg("--max-total-graph-bytes")
                        .arg(
                            config
                                .max_cached_graph_bytes
                                .unwrap_or_default()
                                .to_string(),
                        )
                        .arg("--max-cache-state-bytes-per-request")
                        .arg(config.max_cache_bytes.unwrap_or_default().to_string());
                }
                ArtifactMode::Embedded => {
                    command
                        .args(["embedded-owned-byte-cached-generate", "--experimental"])
                        .arg("--manifest-sha256")
                        .arg(manifest_sha)
                        .arg("--model-sha256")
                        .arg(model_sha)
                        .arg("--max-cache-state-bytes-per-request")
                        .arg(config.max_cache_bytes.unwrap_or_default().to_string());
                }
            }
        }
    }
    command
        .arg("--runtime-config")
        .arg(runtime)
        .arg("--runtime-config-sha256")
        .arg(&service.runtime_sha256)
        .arg("--runtime-config-max-bytes")
        .arg(service.runtime_max_bytes.to_string())
        .arg("--max-new-bytes")
        .arg(config.max_generated_bytes.to_string())
        .env("RAYON_NUM_THREADS", config.worker_threads.to_string())
        .env("OMP_NUM_THREADS", config.worker_threads.to_string())
        .env("OPENBLAS_NUM_THREADS", config.worker_threads.to_string())
        .env_remove("LFG_OWNED_ASSESSMENT_TOKEN")
        .env_remove("LFG_FEEDBACK_TOKEN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    if !service.framing.assistant_suffix.is_empty() {
        command.arg("--stop").arg(&service.framing.assistant_suffix);
    }
    let child_run = run_child(
        command,
        &input,
        Duration::from_millis(config.timeout_ms),
        config.max_worker_output_bytes,
    )
    .await;
    let lifecycle = child_run.lifecycle;
    let raw = match child_run.result {
        Ok(raw) => raw,
        Err(ChildFailure::Spawn) => {
            return add_worker_lifecycle(
                response(
                    "unavailable",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                lifecycle,
            );
        }
        Err(ChildFailure::Timeout) => {
            return add_worker_lifecycle(
                response(
                    "timeout",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                lifecycle,
            );
        }
        Err(ChildFailure::Failed | ChildFailure::OutputLimit) => {
            return add_worker_lifecycle(
                response(
                    "inference_error",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                lifecycle,
            );
        }
    };
    let worker: Value = match serde_json::from_slice::<Value>(&raw) {
        Ok(value) if value["schema_version"] == "lfg.owned-byte-generation.v1" => value,
        _ => {
            return add_worker_lifecycle(
                response(
                    "malformed_output",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                lifecycle,
            );
        }
    };
    let mut text = match worker["text"].as_str() {
        Some(value) if value.len() <= config.max_generated_bytes => value.to_owned(),
        _ => {
            return add_worker_lifecycle(
                response(
                    "malformed_output",
                    Some(service.engine.clone()),
                    started,
                    None,
                    None,
                    None,
                ),
                lifecycle,
            );
        }
    };
    if let Some(suffix) = text.strip_suffix(&service.framing.assistant_suffix) {
        text = suffix.to_owned();
    }
    let stop_reason = worker["stop_reason"].as_str().map(str::to_owned);
    let state = if stop_reason.as_deref() == Some("stop_sequence") {
        "ok"
    } else {
        "truncated"
    };
    let status = if state == "ok" { "ok" } else { "truncated" };
    if status == "truncated" {
        return add_worker_lifecycle(
            response(
                status,
                Some(service.engine.clone()),
                started,
                stop_reason,
                Some(text),
                None,
            ),
            lifecycle,
        );
    }
    let parsed = match serde_json::from_str::<Value>(&text) {
        Ok(value) => match validate_output(value, &config.required_output_fields) {
            Ok(value) => value,
            Err(OutputProblem::Incomplete(value)) => {
                return add_worker_lifecycle(
                    response(
                        "incomplete_output",
                        Some(service.engine.clone()),
                        started,
                        stop_reason,
                        Some(text),
                        Some(value),
                    ),
                    lifecycle,
                );
            }
            Err(OutputProblem::Malformed) => {
                return add_worker_lifecycle(
                    response(
                        "malformed_output",
                        Some(service.engine.clone()),
                        started,
                        stop_reason,
                        Some(text),
                        None,
                    ),
                    lifecycle,
                );
            }
        },
        Err(_) => {
            return add_worker_lifecycle(
                response(
                    "malformed_output",
                    Some(service.engine.clone()),
                    started,
                    stop_reason,
                    Some(text),
                    None,
                ),
                lifecycle,
            );
        }
    };
    add_worker_lifecycle(
        response(
            status,
            Some(service.engine.clone()),
            started,
            stop_reason,
            Some(text),
            Some(parsed),
        ),
        lifecycle,
    )
}

#[derive(Debug, PartialEq)]
enum OutputProblem {
    Incomplete(Value),
    Malformed,
}

fn validate_output(value: Value, required: &[RequiredField]) -> Result<Value, OutputProblem> {
    let object = value.as_object().ok_or(OutputProblem::Malformed)?;
    let mut missing = false;
    for field in required {
        let Some(found) = object.get(&field.name) else {
            missing = true;
            continue;
        };
        let matches = match field.value_type {
            OutputType::String => found.is_string(),
            OutputType::Number => found.is_number(),
            OutputType::Integer => found.as_i64().is_some() || found.as_u64().is_some(),
            OutputType::Boolean => found.is_boolean(),
            OutputType::Array => found.is_array(),
            OutputType::Object => found.is_object(),
        };
        let non_empty = !field.non_empty
            || match field.value_type {
                OutputType::String => !found.as_str().is_some_and(|value| value.trim().is_empty()),
                OutputType::Array => !found.as_array().is_some_and(Vec::is_empty),
                _ => true,
            };
        let integer = found
            .as_i64()
            .map(i128::from)
            .or_else(|| found.as_u64().map(i128::from));
        let in_integer_range = field
            .min_integer
            .is_none_or(|min| integer.is_some_and(|n| n >= i128::from(min)))
            && field
                .max_integer
                .is_none_or(|max| integer.is_some_and(|n| n <= i128::from(max)));
        if !matches || !non_empty || !in_integer_range {
            return Err(OutputProblem::Malformed);
        }
    }
    if missing {
        Err(OutputProblem::Incomplete(value))
    } else {
        Ok(value)
    }
}

fn valid_output_policy(required: &[RequiredField]) -> bool {
    if required.is_empty() {
        return false;
    }
    let mut names = std::collections::BTreeSet::new();
    for field in required {
        if field.name.trim().is_empty() || !names.insert(&field.name) {
            return false;
        }
        if (field.min_integer.is_some() || field.max_integer.is_some())
            && field.value_type != OutputType::Integer
        {
            return false;
        }
        if field
            .min_integer
            .zip(field.max_integer)
            .is_some_and(|(minimum, maximum)| minimum > maximum)
        {
            return false;
        }
        if field.non_empty && !matches!(field.value_type, OutputType::String | OutputType::Array) {
            return false;
        }
    }
    true
}

fn valid_assistant_stop(stop: &str, max_generated_bytes: usize) -> bool {
    stop.len() <= max_generated_bytes
}

#[derive(Debug, PartialEq, Eq)]
enum ChildFailure {
    Spawn,
    Timeout,
    Failed,
    OutputLimit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkerLifecycle {
    started: Option<bool>,
    completed_successfully: Option<bool>,
}

struct ChildRun {
    result: Result<Vec<u8>, ChildFailure>,
    lifecycle: WorkerLifecycle,
}

enum ChildTaskFailure {
    BeforeWait(ChildFailure),
    Exit(ChildFailure, bool),
}

async fn run_child(mut command: Command, input: &[u8], timeout: Duration, cap: usize) -> ChildRun {
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => {
            return ChildRun {
                result: Err(ChildFailure::Spawn),
                lifecycle: WorkerLifecycle {
                    started: Some(false),
                    completed_successfully: Some(false),
                },
            };
        }
    };
    let mut lifecycle = WorkerLifecycle {
        started: Some(true),
        completed_successfully: None,
    };
    let Some(mut stdin) = child.stdin.take() else {
        lifecycle.completed_successfully = terminate(&mut child).await;
        return ChildRun {
            result: Err(ChildFailure::Failed),
            lifecycle,
        };
    };
    let Some(stdout) = child.stdout.take() else {
        lifecycle.completed_successfully = terminate(&mut child).await;
        return ChildRun {
            result: Err(ChildFailure::Failed),
            lifecycle,
        };
    };
    let task = async {
        stdin
            .write_all(input)
            .await
            .map_err(|_| ChildTaskFailure::BeforeWait(ChildFailure::Failed))?;
        stdin
            .shutdown()
            .await
            .map_err(|_| ChildTaskFailure::BeforeWait(ChildFailure::Failed))?;
        // `ChildStdin::poll_shutdown` is a no-op on some platforms. The child
        // worker reads JSONL until EOF, so close the pipe before awaiting stdout.
        drop(stdin);
        let mut output = Vec::with_capacity(cap.min(4096));
        stdout
            .take(cap.saturating_add(1) as u64)
            .read_to_end(&mut output)
            .await
            .map_err(|_| ChildTaskFailure::BeforeWait(ChildFailure::Failed))?;
        if output.len() > cap {
            return Err(ChildTaskFailure::BeforeWait(ChildFailure::OutputLimit));
        }
        let status = child
            .wait()
            .await
            .map_err(|_| ChildTaskFailure::BeforeWait(ChildFailure::Failed))?;
        if !status.success() {
            return Err(ChildTaskFailure::Exit(ChildFailure::Failed, false));
        }
        Ok(output)
    };
    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(output)) => ChildRun {
            result: Ok(output),
            lifecycle: WorkerLifecycle {
                started: Some(true),
                completed_successfully: Some(true),
            },
        },
        Ok(Err(ChildTaskFailure::Exit(reason, completed_successfully))) => ChildRun {
            result: Err(reason),
            lifecycle: WorkerLifecycle {
                started: Some(true),
                completed_successfully: Some(completed_successfully),
            },
        },
        Ok(Err(ChildTaskFailure::BeforeWait(reason))) => {
            lifecycle.completed_successfully = terminate(&mut child).await;
            ChildRun {
                result: Err(reason),
                lifecycle,
            }
        }
        Err(_) => {
            lifecycle.completed_successfully = terminate(&mut child).await;
            ChildRun {
                result: Err(ChildFailure::Timeout),
                lifecycle,
            }
        }
    }
}

async fn terminate(child: &mut Child) -> Option<bool> {
    let _ = child.start_kill();
    child.wait().await.ok().map(|status| status.success())
}

fn response(
    status: &'static str,
    engine: Option<Value>,
    started: Instant,
    stop_reason: Option<String>,
    output_text: Option<String>,
    output: Option<Value>,
) -> (StatusCode, Json<Value>) {
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let engine_id = engine
        .as_ref()
        .and_then(|value| value.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let response = InferenceResponse {
        schema_version: "lfg.owned-assessment.v1",
        status,
        engine,
        elapsed_ms,
        stop_reason: stop_reason.clone(),
        measurement: OwnedRunMeasurement {
            schema_version: "lfg.owned-run-measurement.v1",
            status,
            engine_id,
            stop_reason,
            handler_elapsed_ms: elapsed_ms,
            output_text_utf8_bytes: output_text.as_ref().map(|text| text.len()),
            worker_started: None,
            worker_completed_successfully: None,
        },
        output_text,
        output,
    };
    (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_else(|_| json!({"schemaVersion":"lfg.owned-assessment.v1","status":"inference_error","engine":null,"elapsedMs":0}))))
}

fn add_worker_lifecycle(
    (status, Json(mut value)): (StatusCode, Json<Value>),
    lifecycle: WorkerLifecycle,
) -> (StatusCode, Json<Value>) {
    value["measurement"]["workerStarted"] = json!(lifecycle.started);
    value["measurement"]["workerCompletedSuccessfully"] = json!(lifecycle.completed_successfully);
    (status, Json(value))
}

fn add_context_digest(
    (status, Json(mut value)): (StatusCode, Json<Value>),
    context_sha256: Option<&str>,
) -> (StatusCode, Json<Value>) {
    if let Some(context_sha256) = context_sha256 {
        value["projectContextSha256"] = json!(context_sha256);
    }
    (status, Json(value))
}

fn auth_reason(code: StatusCode) -> &'static str {
    if code == StatusCode::SERVICE_UNAVAILABLE {
        "session_token_unavailable"
    } else {
        "local_access_denied"
    }
}

fn list_env(name: &str, hosts: bool) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .filter(|v| {
            if hosts {
                super::local_access::valid_authority(v)
            } else {
                super::local_access::valid_origin(v)
            }
        })
        .take(16)
        .map(str::to_owned)
        .collect()
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, &'static str> {
    let meta = std::fs::symlink_metadata(path).map_err(|_| "asset_unavailable")?;
    if !meta.file_type().is_file() || meta.len() > limit {
        return Err("asset_unavailable");
    }
    let file = std::fs::File::open(path).map_err(|_| "asset_unavailable")?;
    let mut bytes = Vec::with_capacity(meta.len().min(limit) as usize);
    file.take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| "asset_unavailable")?;
    if bytes.len() as u64 > limit {
        return Err("asset_unavailable");
    }
    Ok(bytes)
}

fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn is_sha256_value(value: &Value) -> bool {
    value.as_str().is_some_and(|value| {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

fn framing_digest(framing: &Framing) -> String {
    let canonical = json!({
        "assistant_prefix": framing.assistant_prefix,
        "assistant_suffix": framing.assistant_suffix,
        "system_prefix": framing.system_prefix,
        "system_suffix": framing.system_suffix,
        "user_prefix": framing.user_prefix,
        "user_suffix": framing.user_suffix,
    });
    digest(&serde_json::to_vec(&canonical).expect("serializing framing fields cannot fail"))
}

fn provenance_links_valid(
    manifest: &Value,
    framing: &Value,
    training: &Value,
    system: &str,
    actual_framing: &Framing,
    training_digest: &str,
) -> bool {
    let system_sha256 = digest(system.as_bytes());
    let training_system_matches = training.get("system_sha256").map_or(true, |value| {
        is_sha256_value(value) && value.as_str() == Some(system_sha256.as_str())
    });
    let training_framing_matches = training.get("framing_sha256").map_or(true, |value| {
        is_sha256_value(value) && value.as_str() == Some(framing_digest(actual_framing).as_str())
    });
    let linked_hashes = [
        &manifest["source_weights_sha256"],
        &manifest["source_manifest_sha256"],
        &framing["system_sha256"],
        &framing["config_sha256"],
        &framing["training_file_sha256"],
        &training["weights_sha256"],
        &training["config_sha256"],
        &training["training_file_sha256"],
    ];
    if linked_hashes.iter().any(|value| !is_sha256_value(value)) {
        return false;
    }
    training_system_matches
        && training_framing_matches
        && framing["system_sha256"].as_str() == Some(system_sha256.as_str())
        && training["weights_sha256"].as_str() == manifest["source_weights_sha256"].as_str()
        && training_digest
            == manifest["source_manifest_sha256"]
                .as_str()
                .unwrap_or_default()
        && framing["config_sha256"].as_str() == training["config_sha256"].as_str()
        && framing["training_file_sha256"].as_str() == training["training_file_sha256"].as_str()
}

fn digest_file(path: &Path, limit: u64) -> Result<(String, u64), &'static str> {
    use std::io::Read;
    let meta = std::fs::symlink_metadata(path).map_err(|_| "asset_unavailable")?;
    if !meta.file_type().is_file() || meta.len() > limit {
        return Err("asset_unavailable");
    }
    let mut file = std::fs::File::open(path).map_err(|_| "asset_unavailable")?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|_| "asset_unavailable")?;
        if count == 0 {
            break;
        }
        total = total.checked_add(count as u64).ok_or("asset_unavailable")?;
        if total > limit {
            return Err("asset_unavailable");
        }
        hasher.update(&buffer[..count]);
    }
    Ok((format!("{:x}", hasher.finalize()), total))
}

fn load_service(config_path: PathBuf) -> Result<Option<Service>, &'static str> {
    let raw = read_bounded(&config_path, MAX_CONFIG_BYTES)?;
    let config: Config = serde_json::from_slice(&raw).map_err(|_| "invalid_configuration")?;
    if config.schema_version != "lfg.owned-assessment-service.v1" {
        return Err("invalid_configuration");
    }
    if !config.enabled {
        return Ok(None);
    }
    if config.max_request_bytes == 0
        || config.max_http_body_bytes == 0
        || config.max_http_body_bytes < config.max_request_bytes
        || config.max_context_bytes == 0
        || config.max_generated_bytes == 0
        || config.max_worker_output_bytes == 0
        || config.max_metadata_bytes == 0
        || config.max_runtime_config_bytes == 0
        || !valid_output_policy(&config.required_output_fields)
        || config.max_worker_output_bytes
            < config
                .max_generated_bytes
                .checked_mul(10)
                .and_then(|n| n.checked_add(1024))
                .unwrap_or(usize::MAX)
        || config.timeout_ms == 0
        || config.max_concurrent == 0
        || config.max_concurrent > Semaphore::MAX_PERMITS
        || config.worker_threads == 0
        || (config.backend == Backend::CachedOnnx
            && (config
                .expected_cached_manifest_sha256
                .as_deref()
                .is_none_or(|hash| {
                    hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
                || config.max_cached_graph_bytes.is_none_or(|bytes| bytes == 0)
                || config.max_cache_bytes.is_none_or(|bytes| bytes == 0)))
    {
        return Err("invalid_configuration");
    }
    let runtime_raw = read_bounded(&config.runtime_config_file, config.max_runtime_config_bytes)?;
    let runtime: RuntimeConfig =
        serde_json::from_slice(&runtime_raw).map_err(|_| "invalid_runtime_configuration")?;
    if runtime.schema_version != "lfg.owned-byte-runtime.v1"
        || runtime.max_manifest_bytes == 0
        || runtime.max_model_bytes == 0
        || runtime.max_context == 0
        || runtime.max_line_bytes == 0
    {
        return Err("invalid_runtime_configuration");
    }
    if config.expected_model_sha256.len() != 64
        || !config
            .expected_model_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("artifact_identity_mismatch");
    }
    let (
        manifest,
        model_sha,
        model_manifest_sha,
        cached_manifest_sha,
        prefill_sha,
        decode_sha,
        context,
    ) = match config.backend {
        Backend::LegacyFullPrefix => {
            if config.artifact_mode != ArtifactMode::Directory {
                return Err("invalid_configuration");
            }
            let artifact = config
                .artifact_directory
                .as_ref()
                .ok_or("invalid_configuration")?;
            let manifest_path = artifact.join("manifest.json");
            let model_path = artifact.join("model.onnx");
            let manifest_raw = read_bounded(&manifest_path, runtime.max_manifest_bytes)?;
            let manifest: Value =
                serde_json::from_slice(&manifest_raw).map_err(|_| "invalid_artifact")?;
            let (model_sha, _) = digest_file(&model_path, runtime.max_model_bytes)?;
            if manifest["schema_version"] != "lfg.owned-byte-onnx.v1"
                || !is_sha256_value(&manifest["model_sha256"])
                || !is_sha256_value(&manifest["source_weights_sha256"])
                || !is_sha256_value(&manifest["source_manifest_sha256"])
                || manifest["model_sha256"].as_str() != Some(model_sha.as_str())
                || model_sha != config.expected_model_sha256
            {
                return Err("artifact_identity_mismatch");
            }
            let manifest_sha = digest(&manifest_raw);
            let context = manifest["context"].clone();
            (
                manifest,
                model_sha,
                manifest_sha,
                None::<String>,
                None::<String>,
                None::<String>,
                context,
            )
        }
        Backend::CachedOnnx => {
            #[cfg(not(feature = "owned-byte-onnx"))]
            {
                return Err("owned_runtime_not_built");
            }
            #[cfg(feature = "owned-byte-onnx")]
            {
                let expected_manifest = config
                    .expected_cached_manifest_sha256
                    .as_deref()
                    .unwrap_or_default();
                let graph_limit = config.max_cached_graph_bytes.unwrap_or_default();
                let cache_limit = config.max_cache_bytes.unwrap_or_default();
                let identity = match config.artifact_mode {
                    ArtifactMode::Directory => {
                        let artifact = config
                            .artifact_directory
                            .as_ref()
                            .ok_or("invalid_configuration")?;
                        super::owned_cached::validate_directory(
                            artifact,
                            expected_manifest,
                            &config.expected_model_sha256,
                            runtime.max_manifest_bytes,
                            runtime.max_context,
                            runtime.max_model_bytes,
                            graph_limit,
                            cache_limit,
                        )
                    }
                    ArtifactMode::Embedded => super::owned_cached::validate_embedded(
                        expected_manifest,
                        &config.expected_model_sha256,
                        runtime.max_manifest_bytes,
                        runtime.max_context,
                        runtime.max_model_bytes,
                        graph_limit,
                        cache_limit,
                    ),
                }
                .map_err(|_| "cached_artifact_invalid")?;
                let manifest = json!({
                    "source_weights_sha256":identity.model_sha256,
                    "source_manifest_sha256":identity.source_manifest_sha256,
                    "context":identity.context
                });
                (
                    manifest,
                    identity.model_sha256,
                    identity.manifest_sha256.clone(),
                    Some(identity.manifest_sha256),
                    Some(identity.prefill_onnx_sha256),
                    Some(identity.decode_onnx_sha256),
                    json!(identity.context),
                )
            }
        }
    };
    let framing_raw = read_bounded(&config.framing_file, config.max_metadata_bytes)?;
    let framing_sha = digest(&framing_raw);
    if framing_sha != config.expected_framing_sha256 {
        return Err("framing_identity_mismatch");
    }
    let framing_json: Value =
        serde_json::from_slice(&framing_raw).map_err(|_| "invalid_framing")?;
    let framing_envelope: FramingEnvelope =
        serde_json::from_value(framing_json.clone()).map_err(|_| "invalid_framing")?;
    if !valid_assistant_stop(
        &framing_envelope.framing.assistant_suffix,
        config.max_generated_bytes,
    ) {
        return Err("runtime_limits_mismatch");
    }
    let training_raw = read_bounded(&config.training_manifest_file, config.max_metadata_bytes)?;
    let training: Value =
        serde_json::from_slice(&training_raw).map_err(|_| "invalid_training_manifest")?;
    if training["schema_version"] != "lfg.owned-assessment-sft-result.v1"
        || framing_json["schema_version"] != "lfg.private-assessment-framing.v1"
        || !provenance_links_valid(
            &manifest,
            &framing_json,
            &training,
            &framing_envelope.system,
            &framing_envelope.framing,
            &digest(&training_raw),
        )
    {
        return Err("provenance_mismatch");
    }
    if runtime.schema_version != "lfg.owned-byte-runtime.v1"
        || config.max_context_bytes > runtime.max_context
        || config.max_context_bytes > manifest["context"].as_u64().unwrap_or(0) as usize
        || runtime.max_generation_bytes.unwrap_or(0) < config.max_generated_bytes
        || runtime.max_line_bytes < config.max_context_bytes.saturating_add(1024)
    {
        return Err("runtime_limits_mismatch");
    }
    let engine = json!({
        "id":if config.backend == Backend::CachedOnnx { "lfg-owned-byte-cached-onnx-experimental" } else { "lfg-owned-byte-onnx-experimental" },
        "inferenceBackend":if config.backend == Backend::CachedOnnx { "cached_onnx" } else { "legacy_full_prefix" },
        "modelIdentityKind":if config.backend == Backend::CachedOnnx { "source_weights_sha256" } else { "onnx_graph_sha256" },
        "modelSha256":model_sha,
        "modelManifestSha256":model_manifest_sha,
        "cachedManifestSha256":cached_manifest_sha,
        "prefillOnnxSha256":prefill_sha,
        "decodeOnnxSha256":decode_sha,
        "sourceWeightsSha256":if config.backend == Backend::CachedOnnx { Some(model_sha.clone()) } else { None },
        "trainingManifestSha256":digest(&training_raw),
        "framingSha256":framing_sha,
        "runtimeConfigSha256":digest(&runtime_raw),
        "serviceConfigSha256":digest(&raw),
        "contextBytes":context
    });
    let runtime_sha256 = digest(&runtime_raw);
    let runtime_max_bytes = config.max_runtime_config_bytes;
    Ok(Some(Service {
        permits: Arc::new(Semaphore::new(config.max_concurrent)),
        backend: config.backend,
        config,
        engine,
        system: framing_envelope.system,
        framing: framing_envelope.framing,
        runtime_line_bytes: runtime.max_line_bytes,
        runtime_sha256,
        runtime_max_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request};
    use tower::ServiceExt;

    fn protected_test_router() -> (Router, SocketAddr, SocketAddr, String) {
        let listener = "127.0.0.1:4317".parse().unwrap();
        let peer = "127.0.0.1:51000".parse().unwrap();
        let token = "0123456789abcdef0123456789abcdef".to_owned();
        let api = Api {
            listener,
            token: Some(Arc::new(token.clone())),
            allowed_hosts: Arc::new(Vec::new()),
            allowed_origins: Arc::new(Vec::new()),
            service: None,
            unavailable_reason: Some("service_disabled"),
        };
        (owned_routes(api), listener, peer, token)
    }

    fn owned_request(
        _listener: SocketAddr,
        peer: SocketAddr,
        path: &str,
        host: &str,
        origin: Option<&str>,
        token: Option<&str>,
        body: &'static str,
    ) -> Request<axum::body::Body> {
        let method = if path.ends_with("/status") {
            "GET"
        } else {
            "POST"
        };
        let mut request = Request::builder()
            .method(method)
            .uri(path)
            .header("host", host)
            .header("content-type", "application/json");
        if let Some(origin) = origin {
            request = request.header("origin", origin);
        }
        if let Some(token) = token {
            request = request.header("x-lfg-session-token", token);
        }
        request
            .extension(ConnectInfo(peer))
            .body(axum::body::Body::from(body))
            .unwrap()
    }

    #[tokio::test]
    async fn protected_routes_are_no_store_and_do_not_echo_private_request_bytes() {
        let (app, listener, peer, token) = protected_test_router();
        let request_body = r#"{"text":"private prompt","context":"private project context","contextSha256":"unused"}"#;
        for (path, body) in [
            ("/api/lfg/assess/owned/status", ""),
            ("/api/lfg/assess/owned", request_body),
        ] {
            let request = owned_request(
                listener,
                peer,
                path,
                &listener.to_string(),
                Some(&format!("http://{listener}")),
                Some(&token),
                body,
            );
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
            let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(!text.contains("private prompt"));
            assert!(!text.contains("private project context"));
            if path.ends_with("/owned") {
                let value: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(value["measurement"]["status"], "unavailable");
                assert_eq!(value["measurement"]["workerStarted"], false);
                assert_eq!(value["measurement"]["workerCompletedSuccessfully"], false);
            }
        }

        let oversized = Request::builder()
            .method("POST")
            .uri("/api/lfg/assess/owned")
            .header("host", listener.to_string())
            .header("content-type", "application/json")
            .header("x-lfg-session-token", token)
            .extension(ConnectInfo(peer))
            .body(axum::body::Body::from(vec![b'x'; 8192]))
            .unwrap();
        let response = app.oneshot(oversized).await.unwrap();
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        assert_ne!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("measurement"));
    }

    #[tokio::test]
    async fn protected_routes_remain_no_store_on_host_origin_token_and_peer_denial() {
        let (app, listener, peer, token) = protected_test_router();
        let authority = listener.to_string();
        let cases = [
            (peer, "127.0.0.1:9999".to_owned(), None, Some(token.clone())),
            (
                peer,
                authority.clone(),
                Some("http://attacker.example".to_owned()),
                Some(token.clone()),
            ),
            (
                peer,
                authority.clone(),
                None,
                Some("wrong-token".to_owned()),
            ),
            (
                "192.0.2.1:51000".parse().unwrap(),
                authority,
                None,
                Some(token.clone()),
            ),
        ];
        for (peer, host, origin, token) in cases {
            let request = owned_request(
                listener,
                peer,
                "/api/lfg/assess/owned/status",
                &host,
                origin.as_deref(),
                token.as_deref(),
                "",
            );
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        }
    }

    #[test]
    fn owned_response_reports_only_observed_timing_and_output_bytes() {
        let (_, Json(value)) = add_worker_lifecycle(
            response(
                "ok",
                Some(json!({"id":"engine"})),
                Instant::now(),
                Some("stop_sequence".into()),
                Some("caf\u{00e9}".into()),
                Some(json!({"intent":"build"})),
            ),
            WorkerLifecycle {
                started: Some(true),
                completed_successfully: Some(true),
            },
        );
        assert_eq!(
            value["measurement"]["schemaVersion"],
            "lfg.owned-run-measurement.v1"
        );
        assert_eq!(value["measurement"]["status"], "ok");
        assert_eq!(value["measurement"]["engineId"], "engine");
        assert_eq!(value["measurement"]["stopReason"], "stop_sequence");
        assert_eq!(value["measurement"]["outputTextUtf8Bytes"], 5);
        assert_eq!(value["measurement"]["workerStarted"], true);
        assert_eq!(value["measurement"]["workerCompletedSuccessfully"], true);
        assert!(value.get("cpuTimeMicros").is_none());
        let diagnostic = value["measurement"].to_string();
        assert!(!diagnostic.contains("private prompt"));
        assert!(!diagnostic.contains("private context"));
        assert!(!diagnostic.contains("caf"));
    }

    #[test]
    fn unavailable_response_measurements_remain_unknown() {
        let (_, Json(value)) = response("inference_error", None, Instant::now(), None, None, None);
        assert_eq!(value["measurement"]["status"], "inference_error");
        assert_eq!(value["measurement"]["engineId"], Value::Null);
        assert_eq!(value["measurement"]["stopReason"], Value::Null);
        assert_eq!(value["measurement"]["outputTextUtf8Bytes"], Value::Null);
        assert_eq!(value["measurement"]["workerStarted"], Value::Null);
        assert_eq!(
            value["measurement"]["workerCompletedSuccessfully"],
            Value::Null
        );
    }

    #[test]
    fn framing_and_request_bounds_count_utf8_bytes() {
        let text = "é";
        assert_eq!(text.len(), 2);
        assert!(
            serde_json::from_slice::<InferRequest>(br#"{"text":"hello","extension":7}"#).is_ok()
        );
    }

    #[test]
    fn assistant_stop_suffix_must_fit_generation_limit() {
        assert!(valid_assistant_stop("END", 3));
        assert!(!valid_assistant_stop("END!", 3));
        assert!(valid_assistant_stop("", 0));
    }

    #[test]
    fn optional_project_context_is_digest_checked_and_rendered_identically() {
        let context = "project: \u{03b1}\nstatus: active";
        let context_digest = digest(context.as_bytes());
        let user =
            render_user_text(Some(context), Some(&context_digest), "fix the route", 128).unwrap();
        assert_eq!(user, "project: \u{03b1}\nstatus: active\n\nfix the route");
        assert_eq!(
            render_user_text(None, None, "fix the route", 128).unwrap(),
            "fix the route"
        );
        assert!(render_user_text(Some(context), None, "fix the route", 128).is_err());
        assert!(
            render_user_text(Some(context), Some(&"0".repeat(64)), "fix the route", 128).is_err()
        );
        assert!(
            render_user_text(Some(context), Some(&context_digest), "fix the parser", 4).is_err()
        );
        assert!(
            serde_json::from_slice::<InferRequest>(
                format!(
                    r#"{{"text":"fix the parser","context":{},"contextSha256":"{}"}}"#,
                    serde_json::to_string(context).unwrap(),
                    context_digest
                )
                .as_bytes()
            )
            .is_ok()
        );
        let user = render_user_text(
            Some("P \u{03b1}"),
            Some(&digest("P \u{03b1}".as_bytes())),
            "Q",
            256,
        )
        .unwrap();
        let framing = Framing {
            system_prefix: "[system]\n".into(),
            system_suffix: "\n".into(),
            user_prefix: "[user]\n".into(),
            user_suffix: "\n".into(),
            assistant_prefix: "[assistant]\n".into(),
            assistant_suffix: "\n".into(),
        };
        let framed = model_prompt("Assess the supplied request.", &framing, &user);
        assert_eq!(
            framed.as_bytes(),
            "[system]\nAssess the supplied request.\n[user]\nP \u{03b1}\n\nQ\n[assistant]\n"
                .as_bytes()
        );
    }

    #[test]
    fn verified_context_digest_is_returned_without_context_text() {
        let context = "private context";
        let context_hash = digest(context.as_bytes());
        let result = add_context_digest(
            response("busy", None, Instant::now(), None, None, None),
            Some(&context_hash),
        );
        assert_eq!(result.1["projectContextSha256"], context_hash);
        assert!(result.1.get("private context").is_none());
        let legacy = add_context_digest(
            response("busy", None, Instant::now(), None, None, None),
            None,
        );
        assert!(legacy.1.get("projectContextSha256").is_none());
    }

    #[test]
    fn output_contract_detects_missing_and_wrong_types_but_keeps_extensions() {
        let required: Vec<RequiredField> = serde_json::from_str(
            r#"[
            {"name":"intent","valueType":"string","nonEmpty":true},
            {"name":"agent_count","valueType":"integer","minInteger":1},
            {"name":"suggested_models","valueType":"array","nonEmpty":true}
        ]"#,
        )
        .unwrap();
        let missing: Value = serde_json::from_str(r#"{"intent":"build"}"#).unwrap();
        assert!(matches!(
            validate_output(missing, &required),
            Err(OutputProblem::Incomplete(_))
        ));
        let wrong_type: Value = serde_json::from_str(
            r#"{"intent":"build","agent_count":"2","suggested_models":["m"]}"#,
        )
        .unwrap();
        assert_eq!(
            validate_output(wrong_type, &required),
            Err(OutputProblem::Malformed)
        );
        let extension: Value = serde_json::from_str(
            r#"{"intent":"build","agent_count":2,"suggested_models":["m"],"custom":{"kept":true}}"#,
        )
        .unwrap();
        assert_eq!(validate_output(extension.clone(), &required), Ok(extension));
        let negative_count: Value =
            serde_json::from_str(r#"{"intent":"build","agent_count":-1,"suggested_models":["m"]}"#)
                .unwrap();
        assert_eq!(
            validate_output(negative_count, &required),
            Err(OutputProblem::Malformed)
        );
        let empty_required: Value =
            serde_json::from_str(r#"{"intent":"","agent_count":1,"suggested_models":[]}"#).unwrap();
        assert_eq!(
            validate_output(empty_required, &required),
            Err(OutputProblem::Malformed)
        );
        let whitespace_required: Value =
            serde_json::from_str(r#"{"intent":"  ","agent_count":1,"suggested_models":["m"]}"#)
                .unwrap();
        assert_eq!(
            validate_output(whitespace_required, &required),
            Err(OutputProblem::Malformed)
        );
        let missing_and_wrong: Value =
            serde_json::from_str(r#"{"intent":"build","suggested_models":"none"}"#).unwrap();
        assert_eq!(
            validate_output(missing_and_wrong, &required),
            Err(OutputProblem::Malformed)
        );
        assert!(valid_output_policy(&required));
        let duplicate: Vec<RequiredField> = serde_json::from_str(
            r#"[{"name":"x","valueType":"string"},{"name":"x","valueType":"array"}]"#,
        )
        .unwrap();
        assert!(!valid_output_policy(&duplicate));
        let invalid_range: Vec<RequiredField> = serde_json::from_str(
            r#"[{"name":"x","valueType":"integer","minInteger":5,"maxInteger":1}]"#,
        )
        .unwrap();
        assert!(!valid_output_policy(&invalid_range));
        let non_object: Value = serde_json::from_str(r#"[]"#).unwrap();
        assert_eq!(
            validate_output(non_object, &required),
            Err(OutputProblem::Malformed)
        );
    }

    #[test]
    fn example_policy_accepts_training_shape_without_inventing_recommendations() {
        let config: Config = serde_json::from_str(include_str!(
            "../config/owned-assessment-service.example.json"
        ))
        .unwrap();
        let proposal = json!({
            "intent": "build",
            "complexity": "unknown",
            "suggested_models": [],
            "agent_count": 0,
            "reasons": ["availability not supplied"],
            "extension": {"preserved": true}
        });
        assert_eq!(
            validate_output(proposal.clone(), &config.required_output_fields),
            Ok(proposal)
        );
    }

    #[test]
    fn provenance_links_reject_missing_null_and_wrong_type_hashes() {
        let weights = "a".repeat(64);
        let config = "b".repeat(64);
        let data = "c".repeat(64);
        let system = "system text";
        let system_sha = digest(system.as_bytes());
        let actual_framing = Framing {
            system_prefix: "<s>".into(),
            system_suffix: "</s>".into(),
            user_prefix: "<u>".into(),
            user_suffix: "</u>".into(),
            assistant_prefix: "<a>".into(),
            assistant_suffix: "</a>".into(),
        };
        let framing_sha = framing_digest(&actual_framing);
        let training = json!({"weights_sha256":weights,"config_sha256":config,
            "training_file_sha256":data,"system_sha256":system_sha,"framing_sha256":framing_sha});
        let raw = serde_json::to_vec(&training).unwrap();
        let manifest =
            json!({"source_weights_sha256":weights,"source_manifest_sha256":digest(&raw)});
        let framing = json!({"system_sha256":digest(system.as_bytes()),"config_sha256":config,"training_file_sha256":data});
        assert!(provenance_links_valid(
            &manifest,
            &framing,
            &training,
            system,
            &actual_framing,
            &digest(&raw)
        ));
        let mut changed_delimiter = actual_framing.clone();
        changed_delimiter.user_prefix = "<different>".into();
        assert!(!provenance_links_valid(
            &manifest,
            &framing,
            &training,
            system,
            &changed_delimiter,
            &digest(&raw)
        ));
        let mut changed_system = training.clone();
        changed_system["system_sha256"] = json!("d".repeat(64));
        let changed_raw = serde_json::to_vec(&changed_system).unwrap();
        let changed_manifest = json!({"source_weights_sha256":weights,
            "source_manifest_sha256":digest(&changed_raw)});
        assert!(!provenance_links_valid(
            &changed_manifest,
            &framing,
            &changed_system,
            system,
            &actual_framing,
            &digest(&changed_raw)
        ));
        for invalid in [Value::Null, json!(7)] {
            let mut bad = training.clone();
            bad["config_sha256"] = invalid;
            assert!(!provenance_links_valid(
                &manifest,
                &framing,
                &bad,
                system,
                &actual_framing,
                &digest(&raw)
            ));
        }
        let mut missing = manifest.clone();
        missing
            .as_object_mut()
            .unwrap()
            .remove("source_weights_sha256");
        assert!(!provenance_links_valid(
            &missing,
            &framing,
            &training,
            system,
            &actual_framing,
            &digest(&raw)
        ));
    }

    #[tokio::test]
    async fn semaphore_enforces_concurrency_without_queueing() {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().try_acquire_owned().unwrap();
        assert!(semaphore.clone().try_acquire_owned().is_err());
        drop(permit);
        assert!(semaphore.try_acquire_owned().is_ok());
    }

    #[tokio::test]
    async fn timeout_kills_and_reaps_worker_child() {
        let executable = std::env::current_exe().unwrap();
        let mut command = Command::new(executable);
        command
            .args([
                "--exact",
                "owned_assessment::tests::sleeping_child_helper",
                "--ignored",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let start = Instant::now();
        let run = run_child(command, b"", Duration::from_millis(200), 4096).await;
        assert_eq!(run.result.unwrap_err(), ChildFailure::Timeout);
        assert_eq!(run.lifecycle.started, Some(true));
        assert_eq!(run.lifecycle.completed_successfully, Some(false));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn spawn_failure_reports_no_worker_started() {
        let missing = std::env::temp_dir().join(format!(
            "lfg-missing-owned-worker-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let run = run_child(Command::new(missing), b"", Duration::from_secs(1), 4096).await;
        assert_eq!(run.result.unwrap_err(), ChildFailure::Spawn);
        assert_eq!(run.lifecycle.started, Some(false));
        assert_eq!(run.lifecycle.completed_successfully, Some(false));
    }

    #[test]
    #[ignore]
    fn sleeping_child_helper() {
        std::thread::sleep(Duration::from_secs(30));
    }

    #[tokio::test]
    async fn worker_output_limit_is_detected_before_waiting_for_exit() {
        let executable = std::env::current_exe().unwrap();
        let mut command = Command::new(executable);
        command
            .args([
                "--exact",
                "owned_assessment::tests::output_child_helper",
                "--ignored",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let run = run_child(command, b"", Duration::from_secs(5), 1).await;
        assert_eq!(run.result.unwrap_err(), ChildFailure::OutputLimit);
        assert_eq!(run.lifecycle.started, Some(true));
        assert_eq!(run.lifecycle.completed_successfully, Some(false));
    }

    #[tokio::test]
    async fn worker_that_reads_to_eof_then_writes_can_exit() {
        let executable = std::env::current_exe().unwrap();
        let mut command = Command::new(executable);
        command
            .args([
                "--exact",
                "owned_assessment::tests::read_to_eof_child_helper",
                "--ignored",
                "--nocapture",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let run = run_child(command, b"worker-input\n", Duration::from_secs(5), 4096).await;
        assert_eq!(run.lifecycle.started, Some(true));
        assert_eq!(run.lifecycle.completed_successfully, Some(true));
        let output = run
            .result
            .expect("worker must observe EOF and produce output");
        // Rust's test harness writes its progress around the helper's stdout;
        // isolate the worker-shaped line emitted by the child test.
        let worker_line = String::from_utf8(output)
            .unwrap()
            .lines()
            .find(|line| line.starts_with('{'))
            .unwrap()
            .to_owned();
        let worker: Value = serde_json::from_str(&worker_line).unwrap();
        assert_eq!(worker["schema_version"], "lfg.owned-byte-generation.v1");
        let text = worker["text"].as_str().unwrap();
        assert_eq!(text, "caf\u{00e9}END");
        let text = text.strip_suffix("END").unwrap().to_owned();
        let (_, Json(result)) = add_worker_lifecycle(
            response(
                "ok",
                Some(json!({"id":"test-engine"})),
                Instant::now(),
                worker["stop_reason"].as_str().map(str::to_owned),
                Some(text),
                Some(json!({"intent":"research"})),
            ),
            run.lifecycle,
        );
        assert_eq!(result["measurement"]["outputTextUtf8Bytes"], 5);
        assert_eq!(result["measurement"]["workerStarted"], true);
        assert_eq!(result["measurement"]["workerCompletedSuccessfully"], true);
    }

    #[tokio::test]
    async fn worker_truncation_and_malformed_output_keep_actual_lifecycle() {
        for (request, expected_text, expected_status) in [
            (
                b"truncated-input\n".as_slice(),
                Some("partial"),
                "truncated",
            ),
            (b"malformed-input\n".as_slice(), None, "malformed_output"),
        ] {
            let executable = std::env::current_exe().unwrap();
            let mut command = Command::new(executable);
            command
                .args([
                    "--exact",
                    "owned_assessment::tests::read_to_eof_child_helper",
                    "--ignored",
                    "--nocapture",
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            let run = run_child(command, request, Duration::from_secs(5), 4096).await;
            assert_eq!(run.lifecycle.started, Some(true));
            assert_eq!(run.lifecycle.completed_successfully, Some(true));
            let output = run.result.unwrap();
            let output = String::from_utf8(output).unwrap();
            let worker_line = output
                .lines()
                .find(|line| line.starts_with('{') || *line == "NOT JSON")
                .unwrap();
            let (text, stop_reason) = if expected_text.is_some() {
                let worker: Value = serde_json::from_str(worker_line).unwrap();
                (
                    worker["text"].as_str().map(str::to_owned),
                    worker["stop_reason"].as_str().map(str::to_owned),
                )
            } else {
                (None, None)
            };
            let result = add_worker_lifecycle(
                response(
                    expected_status,
                    Some(json!({"id":"test-engine"})),
                    Instant::now(),
                    stop_reason,
                    text,
                    None,
                ),
                run.lifecycle,
            );
            let Json(value) = result.1;
            assert_eq!(value["measurement"]["status"], expected_status);
            assert_eq!(
                value["measurement"]["outputTextUtf8Bytes"],
                expected_text.map_or(Value::Null, |text| json!(text.len()))
            );
        }
    }

    #[test]
    #[ignore]
    fn read_to_eof_child_helper() {
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();
        match input.as_slice() {
            b"worker-input\n" => println!(
                "{}",
                json!({
                    "schema_version": "lfg.owned-byte-generation.v1",
                    "text": "caf\u{00e9}END",
                    "stop_reason": "stop_sequence"
                })
            ),
            b"truncated-input\n" => println!(
                "{}",
                json!({
                    "schema_version": "lfg.owned-byte-generation.v1",
                    "text": "partial",
                    "stop_reason": "byte_limit"
                })
            ),
            b"malformed-input\n" => println!("NOT JSON"),
            _ => panic!("unexpected test worker input"),
        }
    }

    #[test]
    #[ignore]
    fn output_child_helper() {
        use std::io::Write;
        let mut stdout = std::io::stdout().lock();
        loop {
            stdout.write_all(&[b'x'; 4096]).unwrap();
            stdout.flush().unwrap();
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn tokenizer_configuration_does_not_report_cpu_isolation() {
        assert!(super::super::local_access::valid_token(
            "0123456789abcdef0123456789abcdef"
        ));
        assert!(!super::super::local_access::valid_token("short"));
    }
}
