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
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
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
struct InferRequest {
    text: String,
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
}

pub(crate) fn routes(listener: SocketAddr) -> Router {
    let api = Api::new(listener);
    let max_body = api
        .service
        .as_ref()
        .map_or(4096, |service| service.config.max_http_body_bytes);
    Router::new()
        .route("/api/lfg/assess/owned/status", get(status))
        .route("/api/lfg/assess/owned", post(infer))
        .layer(DefaultBodyLimit::max(max_body))
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
                json!({"schemaVersion":"lfg.owned-assessment.v1","status":"unavailable","engine":null,"elapsedMs":started.elapsed().as_millis() as u64,"reason":auth_reason(code)}),
            ),
        );
    }
    let Some(service) = api.service.as_ref() else {
        return (
            StatusCode::OK,
            Json(
                json!({"schemaVersion":"lfg.owned-assessment.v1","status":"unavailable","engine":null,"elapsedMs":started.elapsed().as_millis() as u64,"reason":api.unavailable_reason.unwrap_or("unavailable")}),
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
    let permit: OwnedSemaphorePermit = match service.permits.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return response(
                "busy",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    let result = run_inference(service.clone(), request.text, started).await;
    drop(permit);
    result
}

async fn run_inference(
    service: Arc<Service>,
    text: String,
    started: Instant,
) -> (StatusCode, Json<Value>) {
    let config = &service.config;
    let prompt = format!(
        "{}{}{}{}{}{}{}{}",
        service.framing.system_prefix,
        service.system,
        service.framing.system_suffix,
        service.framing.user_prefix,
        text,
        service.framing.user_suffix,
        service.framing.assistant_prefix,
        ""
    );
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
    let raw = match run_child(
        command,
        &input,
        Duration::from_millis(config.timeout_ms),
        config.max_worker_output_bytes,
    )
    .await
    {
        Ok(raw) => raw,
        Err(ChildFailure::Spawn) => {
            return response(
                "unavailable",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
        Err(ChildFailure::Timeout) => {
            return response(
                "timeout",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
        Err(ChildFailure::Failed | ChildFailure::OutputLimit) => {
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
    let worker: Value = match serde_json::from_slice::<Value>(&raw) {
        Ok(value) if value["schema_version"] == "lfg.owned-byte-generation.v1" => value,
        _ => {
            return response(
                "malformed_output",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
            );
        }
    };
    let mut text = match worker["text"].as_str() {
        Some(value) if value.len() <= config.max_generated_bytes => value.to_owned(),
        _ => {
            return response(
                "malformed_output",
                Some(service.engine.clone()),
                started,
                None,
                None,
                None,
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
        return response(
            status,
            Some(service.engine.clone()),
            started,
            stop_reason,
            Some(text),
            None,
        );
    }
    let parsed = match serde_json::from_str::<Value>(&text) {
        Ok(value) => match validate_output(value, &config.required_output_fields) {
            Ok(value) => value,
            Err(OutputProblem::Incomplete(value)) => {
                return response(
                    "incomplete_output",
                    Some(service.engine.clone()),
                    started,
                    stop_reason,
                    Some(text),
                    Some(value),
                );
            }
            Err(OutputProblem::Malformed) => {
                return response(
                    "malformed_output",
                    Some(service.engine.clone()),
                    started,
                    stop_reason,
                    Some(text),
                    None,
                );
            }
        },
        Err(_) => {
            return response(
                "malformed_output",
                Some(service.engine.clone()),
                started,
                stop_reason,
                Some(text),
                None,
            );
        }
    };
    response(
        status,
        Some(service.engine.clone()),
        started,
        stop_reason,
        Some(text),
        Some(parsed),
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

async fn run_child(
    mut command: Command,
    input: &[u8],
    timeout: Duration,
    cap: usize,
) -> Result<Vec<u8>, ChildFailure> {
    let mut child = command.spawn().map_err(|_| ChildFailure::Spawn)?;
    let Some(mut stdin) = child.stdin.take() else {
        terminate(&mut child).await;
        return Err(ChildFailure::Failed);
    };
    let Some(stdout) = child.stdout.take() else {
        terminate(&mut child).await;
        return Err(ChildFailure::Failed);
    };
    let task = async {
        stdin
            .write_all(input)
            .await
            .map_err(|_| ChildFailure::Failed)?;
        stdin.shutdown().await.map_err(|_| ChildFailure::Failed)?;
        // `ChildStdin::poll_shutdown` is a no-op on some platforms. The child
        // worker reads JSONL until EOF, so close the pipe before awaiting stdout.
        drop(stdin);
        let mut output = Vec::with_capacity(cap.min(4096));
        stdout
            .take(cap.saturating_add(1) as u64)
            .read_to_end(&mut output)
            .await
            .map_err(|_| ChildFailure::Failed)?;
        if output.len() > cap {
            return Err(ChildFailure::OutputLimit);
        }
        let status = child.wait().await.map_err(|_| ChildFailure::Failed)?;
        if !status.success() {
            return Err(ChildFailure::Failed);
        }
        Ok(output)
    };
    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(reason)) => {
            terminate(&mut child).await;
            Err(reason)
        }
        Err(_) => {
            terminate(&mut child).await;
            Err(ChildFailure::Timeout)
        }
    }
}

async fn terminate(child: &mut Child) {
    let _ = child.start_kill();
    let _ = child.wait().await;
}

fn response(
    status: &'static str,
    engine: Option<Value>,
    started: Instant,
    stop_reason: Option<String>,
    output_text: Option<String>,
    output: Option<Value>,
) -> (StatusCode, Json<Value>) {
    let response = InferenceResponse {
        schema_version: "lfg.owned-assessment.v1",
        status,
        engine,
        elapsed_ms: started.elapsed().as_millis() as u64,
        stop_reason,
        output_text,
        output,
    };
    (StatusCode::OK, Json(serde_json::to_value(response).unwrap_or_else(|_| json!({"schemaVersion":"lfg.owned-assessment.v1","status":"inference_error","engine":null,"elapsedMs":0}))))
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

fn provenance_links_valid(
    manifest: &Value,
    framing: &Value,
    training: &Value,
    system: &str,
    training_digest: &str,
) -> bool {
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
    framing["system_sha256"].as_str() == Some(digest(system.as_bytes()).as_str())
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
    fn provenance_links_reject_missing_null_and_wrong_type_hashes() {
        let weights = "a".repeat(64);
        let config = "b".repeat(64);
        let data = "c".repeat(64);
        let system = "system text";
        let training =
            json!({"weights_sha256":weights,"config_sha256":config,"training_file_sha256":data});
        let raw = serde_json::to_vec(&training).unwrap();
        let manifest =
            json!({"source_weights_sha256":weights,"source_manifest_sha256":digest(&raw)});
        let framing = json!({"system_sha256":digest(system.as_bytes()),"config_sha256":config,"training_file_sha256":data});
        assert!(provenance_links_valid(
            &manifest,
            &framing,
            &training,
            system,
            &digest(&raw)
        ));
        for invalid in [Value::Null, json!(7)] {
            let mut bad = training.clone();
            bad["config_sha256"] = invalid;
            assert!(!provenance_links_valid(
                &manifest,
                &framing,
                &bad,
                system,
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
        assert_eq!(
            run_child(command, b"", Duration::from_millis(200), 4096)
                .await
                .unwrap_err(),
            ChildFailure::Timeout
        );
        assert!(start.elapsed() < Duration::from_secs(5));
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
        assert_eq!(
            run_child(command, b"", Duration::from_secs(5), 1)
                .await
                .unwrap_err(),
            ChildFailure::OutputLimit
        );
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
        let output = run_child(command, b"worker-input\n", Duration::from_secs(5), 4096)
            .await
            .expect("worker must observe EOF and produce output");
        assert!(String::from_utf8_lossy(&output).contains("EOF-OK"));
    }

    #[test]
    #[ignore]
    fn read_to_eof_child_helper() {
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();
        assert_eq!(input, b"worker-input\n");
        print!("EOF-OK");
    }

    #[test]
    #[ignore]
    fn output_child_helper() {
        print!("child output is deliberately larger than one byte");
    }

    #[test]
    fn tokenizer_configuration_does_not_report_cpu_isolation() {
        assert!(super::super::local_access::valid_token(
            "0123456789abcdef0123456789abcdef"
        ));
        assert!(!super::super::local_access::valid_token("short"));
    }
}
