//! Opt-in assessment through a chat model on a loopback port, for the owner's dev version.
//! The model is an open model and is labelled as one, never as LFG's own. Private prompts
//! and generated text are handled in memory only and are never logged here.
use std::{
    net::SocketAddr,
    path::PathBuf,
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
use http_body_util::{BodyExt, Full};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use lfg_core::assessment::KnownModel;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use crate::local_access::{self, LocalAccessError};
use crate::owned_assessment::{
    InferRequest, OutputProblem, RequiredField, auth_reason, digest, list_env, read_bounded,
    render_user_text, valid_output_policy, validate_output,
};

const CONFIG_ENV: &str = "LFG_DEV_MODEL_CONFIG";
const TOKEN_ENV: &str = "LFG_DEV_MODEL_TOKEN";
const MAX_CONFIG_BYTES: u64 = 128 * 1024;
const RESULT_SCHEMA: &str = "lfg.dev-model-assessment.v1";
const STATUS_SCHEMA: &str = "lfg.dev-model-status.v1";
/// Replaced in the instruction by the models LFG currently knows are available.
const MODEL_LIST_SLOT: &str = "{{available_models}}";
/// Request fields the service owns, so a configured extension cannot change them.
const RESERVED_EXTENSIONS: [&str; 6] = [
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "response_format",
    "stream",
];

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    schema_version: String,
    enabled: bool,
    endpoint: SocketAddr,
    served_model: String,
    model_label: String,
    weights_sha256: String,
    api_key_env: String,
    instruction_file: PathBuf,
    expected_instruction_sha256: String,
    max_instruction_bytes: u64,
    #[serde(default)]
    request_extensions: Map<String, Value>,
    required_output_fields: Vec<RequiredField>,
    max_http_body_bytes: usize,
    max_request_bytes: usize,
    max_context_bytes: usize,
    max_generated_tokens: u32,
    temperature: f64,
    max_response_bytes: usize,
    timeout_ms: u64,
    max_concurrent: usize,
}

struct Service {
    config: Config,
    engine: Value,
    instruction: String,
    api_key: String,
    client: Client<HttpConnector, Full<Bytes>>,
    permits: Arc<Semaphore>,
}

pub(crate) type KnownModels = Arc<dyn Fn() -> Vec<KnownModel> + Send + Sync>;

#[derive(Clone)]
struct Api {
    listener: SocketAddr,
    known_models: KnownModels,
    token: Option<Arc<String>>,
    allowed_hosts: Arc<Vec<String>>,
    allowed_origins: Arc<Vec<String>>,
    service: Option<Arc<Service>>,
    unavailable_reason: Option<&'static str>,
}

pub(crate) fn routes(listener: SocketAddr, known_models: KnownModels) -> Router {
    router(Api::new(listener, known_models))
}

fn router(api: Api) -> Router {
    let max_body = api
        .service
        .as_ref()
        .map_or(4096, |service| service.config.max_http_body_bytes);
    Router::new()
        .route("/api/lfg/assess/dev-model/status", get(status))
        .route("/api/lfg/assess/dev-model", post(infer))
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(api)
}

impl Api {
    fn new(listener: SocketAddr, known_models: KnownModels) -> Self {
        let token = std::env::var(TOKEN_ENV)
            .ok()
            .filter(|value| local_access::valid_token(value))
            .map(Arc::new);
        let (service, unavailable_reason) = if !listener.ip().is_loopback() {
            (None, Some("loopback_listener_required"))
        } else if token.is_none() {
            (None, Some("session_token_unavailable"))
        } else {
            match std::env::var_os(CONFIG_ENV) {
                None => (None, Some("service_disabled")),
                Some(path) => {
                    match load_service(PathBuf::from(path), |name| std::env::var(name).ok()) {
                        Ok(Some(service)) => (Some(Arc::new(service)), None),
                        Ok(None) => (None, Some("service_disabled")),
                        Err(reason) => (None, Some(reason)),
                    }
                }
            }
        };
        Self {
            listener,
            known_models,
            token,
            allowed_hosts: Arc::new(list_env("LFG_DEV_MODEL_ALLOWED_HOSTS", true)),
            allowed_origins: Arc::new(list_env("LFG_DEV_MODEL_ALLOWED_ORIGINS", false)),
            service,
            unavailable_reason,
        }
    }

    fn authorize(&self, peer: SocketAddr, headers: &HeaderMap) -> Result<(), StatusCode> {
        local_access::authorize(
            self.listener,
            peer,
            headers,
            self.token.as_deref().map(String::as_str),
            &self.allowed_hosts,
            &self.allowed_origins,
        )
        .map_err(|error| match error {
            LocalAccessError::TokenUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::FORBIDDEN,
        })
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn load_service(
    config_path: PathBuf,
    environment: impl Fn(&str) -> Option<String>,
) -> Result<Option<Service>, &'static str> {
    let raw = read_bounded(&config_path, MAX_CONFIG_BYTES)?;
    let config: Config = serde_json::from_slice(&raw).map_err(|_| "invalid_configuration")?;
    if config.schema_version != "lfg.dev-model-service.v1" {
        return Err("invalid_configuration");
    }
    if !config.enabled {
        return Ok(None);
    }
    if !config.endpoint.ip().is_loopback()
        || config.endpoint.port() == 0
        || config.served_model.trim().is_empty()
        || config.model_label.trim().is_empty()
        || config.api_key_env.trim().is_empty()
        || !is_sha256(&config.weights_sha256)
        || !is_sha256(&config.expected_instruction_sha256)
        || config.max_instruction_bytes == 0
        || config.max_request_bytes == 0
        || config.max_http_body_bytes < config.max_request_bytes
        || config.max_context_bytes == 0
        || config.max_generated_tokens == 0
        || !config.temperature.is_finite()
        || !(0.0..=2.0).contains(&config.temperature)
        || config.max_response_bytes == 0
        || config.timeout_ms == 0
        || config.max_concurrent == 0
        || config.max_concurrent > Semaphore::MAX_PERMITS
        || !valid_output_policy(&config.required_output_fields)
        || RESERVED_EXTENSIONS
            .iter()
            .any(|name| config.request_extensions.contains_key(*name))
    {
        return Err("invalid_configuration");
    }
    let api_key = environment(&config.api_key_env)
        .filter(|key| !key.is_empty() && !key.contains(['\r', '\n']))
        .ok_or("model_key_unavailable")?;
    let instruction_raw = read_bounded(&config.instruction_file, config.max_instruction_bytes)?;
    let instruction_sha256 = digest(&instruction_raw);
    if instruction_sha256 != config.expected_instruction_sha256 {
        return Err("instruction_identity_mismatch");
    }
    let instruction = String::from_utf8(instruction_raw)
        .ok()
        .filter(|text| !text.trim().is_empty())
        .ok_or("instruction_identity_mismatch")?;
    let engine = json!({
        "id":"lfg-dev-open-model",
        "kind":"open_model",
        "label":config.model_label,
        "servedModel":config.served_model,
        "modelIdentityKind":"operator_stated_weights_sha256",
        "modelSha256":config.weights_sha256,
        "instructionSha256":instruction_sha256,
        "serviceConfigSha256":digest(&raw),
    });
    Ok(Some(Service {
        permits: Arc::new(Semaphore::new(config.max_concurrent)),
        client: Client::builder(TokioExecutor::new()).build_http(),
        config,
        engine,
        instruction,
        api_key,
    }))
}

async fn status(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(code) = api.authorize(peer, &headers) {
        return (
            code,
            Json(json!({
                "schemaVersion":STATUS_SCHEMA,"enabled":false,"status":"unavailable",
                "engine":null,"reason":auth_reason(code),
            })),
        );
    }
    let service = api.service.as_ref();
    let state = if service.is_some() {
        "ready"
    } else if api.unavailable_reason == Some("service_disabled") {
        "disabled"
    } else {
        "unavailable"
    };
    (
        StatusCode::OK,
        Json(json!({
            "schemaVersion":STATUS_SCHEMA,
            "enabled":service.is_some(),
            "status":state,
            "engine":service.map(|s| s.engine.clone()),
            "limits":service.map(|s| json!({
                "maxRequestBytes":s.config.max_request_bytes,
                "maxHttpBodyBytes":s.config.max_http_body_bytes,
                "maxContextBytes":s.config.max_context_bytes,
                "maxGeneratedTokens":s.config.max_generated_tokens,
                "timeoutMs":s.config.timeout_ms,
                "maxConcurrent":s.config.max_concurrent,
            })),
            "reason":if service.is_none() { api.unavailable_reason } else { None },
        })),
    )
}

struct Outcome {
    status: &'static str,
    stop_reason: Option<String>,
    output_text: Option<String>,
    output: Option<Value>,
}

impl Outcome {
    fn only(status: &'static str) -> Self {
        Self {
            status,
            stop_reason: None,
            output_text: None,
            output: None,
        }
    }

    fn into_response(
        self,
        engine: Option<Value>,
        started: Instant,
        context_sha256: Option<&str>,
    ) -> (StatusCode, Json<Value>) {
        let mut body = json!({
            "schemaVersion":RESULT_SCHEMA,
            "status":self.status,
            "engine":engine,
            "elapsedMs":started.elapsed().as_millis() as u64,
        });
        if let Some(value) = self.stop_reason {
            body["stopReason"] = json!(value);
        }
        if let Some(value) = self.output_text {
            body["outputText"] = json!(value);
        }
        if let Some(value) = self.output {
            body["output"] = value;
        }
        if let Some(value) = context_sha256 {
            body["projectContextSha256"] = json!(value);
        }
        (StatusCode::OK, Json(body))
    }
}

async fn infer(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let started = Instant::now();
    if let Err(code) = api.authorize(peer, &headers) {
        let (_, Json(mut value)) = Outcome::only("unavailable").into_response(None, started, None);
        value["reason"] = json!(auth_reason(code));
        return (code, Json(value));
    }
    let Some(service) = api.service.as_ref() else {
        let (code, Json(mut value)) =
            Outcome::only("unavailable").into_response(None, started, None);
        value["reason"] = json!(api.unavailable_reason.unwrap_or("unavailable"));
        return (code, Json(value));
    };
    let engine = Some(service.engine.clone());
    let config = &service.config;
    let request = serde_json::from_slice::<InferRequest>(&body)
        .ok()
        .filter(|request| {
            body.len() <= config.max_http_body_bytes
                && !request.text.is_empty()
                && request.text.len() <= config.max_request_bytes
        });
    let Some(request) = request else {
        return Outcome::only("invalid_input").into_response(engine, started, None);
    };
    let Ok(user_text) = render_user_text(
        request.context.as_deref(),
        request.context_sha256.as_deref(),
        &request.text,
        config.max_context_bytes,
    ) else {
        return Outcome::only("invalid_input").into_response(engine, started, None);
    };
    // The context bytes have now been checked against this digest.
    let context_sha256 = request.context_sha256.as_deref();
    let models = (api.known_models)();
    let listed: Vec<&KnownModel> = models.iter().filter(|model| model.available).collect();
    let uses_list = service.instruction.contains(MODEL_LIST_SLOT);
    let model_list = render_model_list(&listed);
    let instruction = service.instruction.replace(MODEL_LIST_SLOT, &model_list);
    if instruction.len().saturating_add(user_text.len()) > config.max_context_bytes {
        return Outcome::only("context_limit").into_response(engine, started, context_sha256);
    }
    let Ok(_permit) = service.permits.clone().try_acquire_owned() else {
        return Outcome::only("busy").into_response(engine, started, context_sha256);
    };
    let mut outcome = match service.ask(&instruction, &user_text).await {
        Ok((text, finish)) => interpret(text, finish, &config.required_output_fields),
        Err(status) => Outcome::only(status),
    };
    if uses_list && let Some(output) = outcome.output.as_mut() {
        keep_listed_models(output, &listed);
    }
    let (code, Json(mut value)) = outcome.into_response(engine, started, context_sha256);
    if uses_list {
        value["modelListSha256"] = json!(digest(model_list.as_bytes()));
    }
    (code, Json(value))
}

fn render_model_list(models: &[&KnownModel]) -> String {
    if models.is_empty() {
        return "unknown".to_owned();
    }
    let rows: Vec<String> = models
        .iter()
        .map(|model| {
            let role = model.role.as_deref().unwrap_or("unknown role");
            format!("{} ({role})", model.model)
        })
        .collect();
    rows.join("; ")
}

/// A name the model made up must not read as a suggestion. It stays visible as its
/// own field, and the raw answer stays in `outputText`.
fn keep_listed_models(output: &mut Value, listed: &[&KnownModel]) {
    let Some(names) = output
        .get_mut("suggested_models")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    let (kept, unlisted): (Vec<Value>, Vec<Value>) =
        std::mem::take(names).into_iter().partition(|name| {
            name.as_str()
                .is_some_and(|name| listed.iter().any(|model| model.model == name))
        });
    *names = kept;
    if !unlisted.is_empty() {
        output["unlisted_models"] = Value::Array(unlisted);
    }
}

impl Service {
    async fn ask(
        &self,
        instruction: &str,
        user_text: &str,
    ) -> Result<(String, Option<String>), &'static str> {
        let config = &self.config;
        let mut body = config.request_extensions.clone();
        body.insert("model".into(), json!(config.served_model));
        body.insert(
            "messages".into(),
            json!([
                {"role":"system","content":instruction},
                {"role":"user","content":user_text},
            ]),
        );
        body.insert("temperature".into(), json!(config.temperature));
        body.insert("max_tokens".into(), json!(config.max_generated_tokens));
        body.insert("response_format".into(), json!({"type":"json_object"}));
        body.insert("stream".into(), json!(false));
        let payload = serde_json::to_vec(&body).map_err(|_| "inference_error")?;
        let request =
            hyper::Request::post(format!("http://{}/v1/chat/completions", config.endpoint))
                .header(hyper::header::CONTENT_TYPE, "application/json")
                .header(
                    hyper::header::AUTHORIZATION,
                    format!("Bearer {}", self.api_key),
                )
                .body(Full::new(Bytes::from(payload)))
                .map_err(|_| "inference_error")?;
        let call = async {
            let mut response = self
                .client
                .request(request)
                .await
                .map_err(|_| "unavailable")?;
            if !response.status().is_success() {
                return Err("inference_error");
            }
            let mut bytes = Vec::new();
            while let Some(frame) = response.body_mut().frame().await {
                let frame = frame.map_err(|_| "inference_error")?;
                if let Some(chunk) = frame.data_ref() {
                    if chunk.len() > config.max_response_bytes.saturating_sub(bytes.len()) {
                        return Err("inference_error");
                    }
                    bytes.extend_from_slice(chunk);
                }
            }
            Ok(bytes)
        };
        let bytes = tokio::time::timeout(Duration::from_millis(config.timeout_ms), call)
            .await
            .map_err(|_| "timeout")??;
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| "malformed_output")?;
        let choice = &value["choices"][0];
        let text = choice["message"]["content"]
            .as_str()
            .ok_or("malformed_output")?;
        Ok((
            text.to_owned(),
            choice["finish_reason"].as_str().map(str::to_owned),
        ))
    }
}

fn interpret(text: String, finish: Option<String>, required: &[RequiredField]) -> Outcome {
    let (status, output) = match finish.as_deref() {
        Some("length") => ("truncated", None),
        Some("stop") => match serde_json::from_str::<Value>(&text) {
            Ok(value) => match validate_output(value, required) {
                Ok(value) => ("ok", Some(value)),
                Err(OutputProblem::Incomplete(value)) => ("incomplete_output", Some(value)),
                Err(OutputProblem::Malformed) => ("malformed_output", None),
            },
            Err(_) => ("malformed_output", None),
        },
        _ => ("malformed_output", None),
    };
    Outcome {
        status,
        stop_reason: finish,
        output_text: Some(text),
        output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::connect_info::MockConnectInfo,
        http::{Request, header},
    };
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;

    const TOKEN: &str = "0123456789abcdefghijklmnopqrstuvwxyz";
    const INSTRUCTION: &str = "Assess this request for routing.";
    const GOOD: &str = r#"{"intent":"build","complexity":"high","suggested_models":["a"],"agent_count":2,"reasons":["many steps"],"extra":"kept"}"#;

    fn sha(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    fn configuration(endpoint: SocketAddr) -> Value {
        json!({
            "schemaVersion":"lfg.dev-model-service.v1",
            "enabled":true,
            "endpoint":endpoint.to_string(),
            "servedModel":"small-model",
            "modelLabel":"Small model (open model, dev only)",
            "weightsSha256":sha("weights"),
            "apiKeyEnv":"TEST_MODEL_KEY",
            "instructionFile":"",
            "expectedInstructionSha256":sha(INSTRUCTION),
            "maxInstructionBytes":4096,
            "requestExtensions":{"chat_template_kwargs":{"enable_thinking":false}},
            "requiredOutputFields":[
                {"name":"intent","valueType":"string","nonEmpty":true},
                {"name":"agent_count","valueType":"integer","minInteger":1},
                {"name":"reasons","valueType":"array","nonEmpty":true},
            ],
            "maxHttpBodyBytes":8192,
            "maxRequestBytes":2048,
            "maxContextBytes":4096,
            "maxGeneratedTokens":512,
            "temperature":0.0,
            "maxResponseBytes":65536,
            "timeoutMs":2000,
            "maxConcurrent":1,
        })
    }

    fn load(mut value: Value) -> Result<Option<Service>, &'static str> {
        let directory = tempfile::tempdir().unwrap();
        let instruction = directory.path().join("instruction.txt");
        std::fs::write(&instruction, INSTRUCTION).unwrap();
        value["instructionFile"] = json!(instruction);
        let path = directory.path().join("config.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        load_service(path, |name| {
            (name == "TEST_MODEL_KEY").then(|| "model-key".to_owned())
        })
    }

    /// A fake chat endpoint that records the last request and returns a fixed reply.
    async fn model(
        status: u16,
        reply: Value,
        delay: Duration,
    ) -> (SocketAddr, Arc<std::sync::Mutex<Option<(String, Value)>>>) {
        let seen = Arc::new(std::sync::Mutex::new(None));
        let recorded = seen.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let reply = reply.clone();
                let recorded = recorded.clone();
                async move {
                    let key = headers[header::AUTHORIZATION].to_str().unwrap().to_owned();
                    *recorded.lock().unwrap() = Some((key, body));
                    tokio::time::sleep(delay).await;
                    (StatusCode::from_u16(status).unwrap(), Json(reply))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        (address, seen)
    }

    fn answer(content: &str, finish: &str) -> Value {
        json!({"choices":[{"message":{"content":content},"finish_reason":finish}]})
    }

    fn app(service: Service) -> Router {
        app_with_models(service, Vec::new())
    }

    fn app_with_models(service: Service, models: Vec<KnownModel>) -> Router {
        let listener: SocketAddr = "127.0.0.1:4317".parse().unwrap();
        router(Api {
            listener,
            known_models: Arc::new(move || models.clone()),
            token: Some(Arc::new(TOKEN.to_owned())),
            allowed_hosts: Arc::new(Vec::new()),
            allowed_origins: Arc::new(Vec::new()),
            service: Some(Arc::new(service)),
            unavailable_reason: None,
        })
        .layer(MockConnectInfo(listener))
    }

    fn post_request(body: &Value, token: &str) -> Request<Body> {
        Request::post("/api/lfg/assess/dev-model")
            .header(header::HOST, "127.0.0.1:4317")
            .header("x-lfg-session-token", token)
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    async fn send(router: Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = router.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn assess(reply: Value) -> Value {
        let (address, _) = model(200, reply, Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        send(
            app(service),
            post_request(&json!({"text":"Add a route"}), TOKEN),
        )
        .await
        .1
    }

    #[tokio::test]
    async fn complete_answer_is_returned_and_labelled_as_an_open_model() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        let (code, body) = send(
            app(service),
            post_request(&json!({"text":"Add a route","future":"ignored"}), TOKEN),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["output"]["agent_count"], 2);
        assert_eq!(body["output"]["extra"], "kept");
        assert_eq!(body["engine"]["id"], "lfg-dev-open-model");
        assert_eq!(body["engine"]["kind"], "open_model");
        assert_eq!(
            body["engine"]["modelIdentityKind"],
            "operator_stated_weights_sha256"
        );
        let (key, request) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(key, "Bearer model-key");
        assert_eq!(request["model"], "small-model");
        assert_eq!(request["messages"][0]["content"], INSTRUCTION);
        assert_eq!(request["messages"][1]["content"], "Add a route");
        assert_eq!(request["max_tokens"], 512);
        assert_eq!(request["response_format"]["type"], "json_object");
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[tokio::test]
    async fn verified_context_reaches_the_model_and_only_its_digest_comes_back() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        let context = "Available: a, b";
        let body = json!({"text":"Add a route","context":context,"contextSha256":sha(context)});
        let (_, reply) = send(app(service), post_request(&body, TOKEN)).await;
        assert_eq!(reply["status"], "ok");
        assert_eq!(reply["projectContextSha256"], sha(context));
        assert!(!reply.to_string().contains(context));
        let (_, request) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(
            request["messages"][1]["content"],
            "Available: a, b\n\nAdd a route"
        );
    }

    #[tokio::test]
    async fn context_with_the_wrong_digest_is_rejected_before_the_model() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        let body = json!({"text":"Add a route","context":"text","contextSha256":sha("other")});
        let (_, reply) = send(app(service), post_request(&body, TOKEN)).await;
        assert_eq!(reply["status"], "invalid_input");
        assert!(seen.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn answers_that_do_not_meet_the_contract_are_not_reported_as_ok() {
        let partial = assess(answer(r#"{"intent":"build"}"#, "stop")).await;
        assert_eq!(partial["status"], "incomplete_output");
        assert_eq!(partial["output"]["intent"], "build");

        let wrong = r#"{"intent":"build","agent_count":0,"reasons":["x"]}"#;
        let malformed = assess(answer(wrong, "stop")).await;
        assert_eq!(malformed["status"], "malformed_output");
        assert_eq!(malformed["outputText"], wrong);
        assert!(malformed.get("output").is_none());

        let invalid = assess(answer("not json", "stop")).await;
        assert_eq!(invalid["status"], "malformed_output");

        let cut = assess(answer(r#"{"intent":"bui"#, "length")).await;
        assert_eq!(cut["status"], "truncated");
        assert_eq!(cut["stopReason"], "length");
        assert_eq!(cut["outputText"], r#"{"intent":"bui"#);
    }

    #[tokio::test]
    async fn model_failures_have_distinct_statuses() {
        let (address, _) = model(500, json!({}), Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        let body = json!({"text":"Add a route"});
        assert_eq!(
            send(app(service), post_request(&body, TOKEN)).await.1["status"],
            "inference_error"
        );

        let (address, _) = model(200, answer(GOOD, "stop"), Duration::from_secs(30)).await;
        let mut config = configuration(address);
        config["timeoutMs"] = json!(50);
        let service = load(config).unwrap().unwrap();
        assert_eq!(
            send(app(service), post_request(&body, TOKEN)).await.1["status"],
            "timeout"
        );

        let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = closed.local_addr().unwrap();
        drop(closed);
        // A refused connection takes about two seconds on Windows.
        let mut config = configuration(address);
        config["timeoutMs"] = json!(30_000);
        let service = load(config).unwrap().unwrap();
        assert_eq!(
            send(app(service), post_request(&body, TOKEN)).await.1["status"],
            "unavailable"
        );

        let (address, _) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let mut config = configuration(address);
        config["maxResponseBytes"] = json!(8);
        let service = load(config).unwrap().unwrap();
        assert_eq!(
            send(app(service), post_request(&body, TOKEN)).await.1["status"],
            "inference_error"
        );
    }

    #[tokio::test]
    async fn limits_stop_requests_before_the_model() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let mut config = configuration(address);
        config["maxRequestBytes"] = json!(8);
        let service = load(config).unwrap().unwrap();
        let (_, reply) = send(
            app(service),
            post_request(&json!({"text":"too long a request"}), TOKEN),
        )
        .await;
        assert_eq!(reply["status"], "invalid_input");

        let mut config = configuration(address);
        config["maxContextBytes"] = json!(INSTRUCTION.len() + 2);
        let service = load(config).unwrap().unwrap();
        let (_, reply) = send(
            app(service),
            post_request(&json!({"text":"too long"}), TOKEN),
        )
        .await;
        assert_eq!(reply["status"], "context_limit");

        let service = load(configuration(address)).unwrap().unwrap();
        let held = service.permits.clone().try_acquire_owned().unwrap();
        let (_, reply) = send(app(service), post_request(&json!({"text":"Add"}), TOKEN)).await;
        assert_eq!(reply["status"], "busy");
        drop(held);
        assert!(seen.lock().unwrap().is_none());
    }

    #[tokio::test]
    async fn access_needs_the_token_and_a_ready_status_names_the_model() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let service = load(configuration(address)).unwrap().unwrap();
        let router = app(service);
        let (code, reply) = send(
            router.clone(),
            post_request(&json!({"text":"Add"}), "wrong-token"),
        )
        .await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        assert_eq!(reply["engine"], Value::Null);
        assert!(seen.lock().unwrap().is_none());

        let status_request = |token: &str| {
            Request::get("/api/lfg/assess/dev-model/status")
                .header(header::HOST, "127.0.0.1:4317")
                .header("x-lfg-session-token", token)
                .body(Body::empty())
                .unwrap()
        };
        let (code, _) = send(router.clone(), status_request("wrong-token")).await;
        assert_eq!(code, StatusCode::FORBIDDEN);
        let (code, reply) = send(router, status_request(TOKEN)).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(reply["status"], "ready");
        assert_eq!(
            reply["engine"]["label"],
            "Small model (open model, dev only)"
        );
        assert_eq!(reply["limits"]["maxGeneratedTokens"], 512);
        assert!(!reply.to_string().contains("instruction.txt"));
    }

    fn known(model: &str, role: Option<&str>, description: &str, available: bool) -> KnownModel {
        KnownModel {
            model: model.to_owned(),
            role: role.map(str::to_owned),
            description: description.to_owned(),
            available,
            ..KnownModel::default()
        }
    }

    fn load_with_instruction(endpoint: SocketAddr, instruction: &str) -> Service {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("instruction.txt");
        std::fs::write(&file, instruction).unwrap();
        let mut value = configuration(endpoint);
        value["instructionFile"] = json!(file);
        value["expectedInstructionSha256"] = json!(sha(instruction));
        let path = directory.path().join("config.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        load_service(path, |name| {
            (name == "TEST_MODEL_KEY").then(|| "model-key".to_owned())
        })
        .unwrap()
        .unwrap()
    }

    #[tokio::test]
    async fn the_live_model_list_fills_the_instruction_and_made_up_names_are_set_aside() {
        let reply = r#"{"intent":"build","agent_count":2,"reasons":["x"],"suggested_models":["real-fast","made-up",7,"gone"]}"#;
        let (address, seen) = model(200, answer(reply, "stop"), Duration::ZERO).await;
        let service = load_with_instruction(address, "Choose from: {{available_models}}. Done.");
        let models = vec![
            known("real-fast", Some("fast"), "Quick work.", true),
            known("no-details", None, "", true),
            known("gone", Some("standard"), "No longer offered.", false),
        ];
        let (_, body) = send(
            app_with_models(service, models),
            post_request(&json!({"text":"Add a route"}), TOKEN),
        )
        .await;
        let (_, request) = seen.lock().unwrap().clone().unwrap();
        let list = "real-fast (fast); no-details (unknown role)";
        assert_eq!(
            request["messages"][0]["content"],
            format!("Choose from: {list}. Done.")
        );
        assert_eq!(body["status"], "ok");
        assert_eq!(body["modelListSha256"], sha(list));
        assert_eq!(body["output"]["suggested_models"], json!(["real-fast"]));
        assert_eq!(
            body["output"]["unlisted_models"],
            json!(["made-up", 7, "gone"])
        );
        assert_eq!(body["outputText"], reply);
    }

    #[tokio::test]
    async fn without_a_known_list_no_name_counts_as_a_suggestion() {
        let reply =
            r#"{"intent":"build","agent_count":1,"reasons":["x"],"suggested_models":["guess"]}"#;
        let (address, seen) = model(200, answer(reply, "stop"), Duration::ZERO).await;
        let service = load_with_instruction(address, "Models: {{available_models}}");
        let (_, body) = send(app(service), post_request(&json!({"text":"Add"}), TOKEN)).await;
        let (_, request) = seen.lock().unwrap().clone().unwrap();
        assert_eq!(request["messages"][0]["content"], "Models: unknown");
        assert_eq!(body["output"]["suggested_models"], json!([]));
        assert_eq!(body["output"]["unlisted_models"], json!(["guess"]));
    }

    #[tokio::test]
    async fn an_instruction_without_the_slot_leaves_the_answer_alone() {
        let body = assess(answer(GOOD, "stop")).await;
        assert_eq!(body["output"]["suggested_models"], json!(["a"]));
        assert!(body.get("modelListSha256").is_none());
    }

    #[tokio::test]
    async fn the_filled_list_counts_towards_the_context_limit() {
        let (address, seen) = model(200, answer(GOOD, "stop"), Duration::ZERO).await;
        let mut value = configuration(address);
        value["maxContextBytes"] = json!(64);
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("instruction.txt");
        let instruction = "{{available_models}}";
        std::fs::write(&file, instruction).unwrap();
        value["instructionFile"] = json!(file);
        value["expectedInstructionSha256"] = json!(sha(instruction));
        let path = directory.path().join("config.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let service = load_service(path, |_| Some("model-key".to_owned()))
            .unwrap()
            .unwrap();
        let models = vec![known(&"long-name-".repeat(40), Some("fast"), "", true)];
        let (_, body) = send(
            app_with_models(service, models),
            post_request(&json!({"text":"Add"}), TOKEN),
        )
        .await;
        assert_eq!(body["status"], "context_limit");
        assert!(seen.lock().unwrap().is_none());
    }

    #[test]
    fn example_configuration_is_valid_and_disabled() {
        let config: Config =
            serde_json::from_str(include_str!("../config/dev-model-service.example.json")).unwrap();
        assert_eq!(config.schema_version, "lfg.dev-model-service.v1");
        assert!(!config.enabled);
        assert!(config.endpoint.ip().is_loopback());
        assert!(valid_output_policy(&config.required_output_fields));
    }

    #[test]
    fn configuration_is_checked_before_use() {
        let endpoint: SocketAddr = "127.0.0.1:18092".parse().unwrap();
        assert!(load(configuration(endpoint)).unwrap().is_some());

        let mut disabled = configuration(endpoint);
        disabled["enabled"] = json!(false);
        assert!(load(disabled).unwrap().is_none());

        let remote = configuration("192.168.1.20:18092".parse().unwrap());
        assert_eq!(load(remote).err(), Some("invalid_configuration"));

        for name in RESERVED_EXTENSIONS {
            let mut reserved = configuration(endpoint);
            reserved["requestExtensions"][name] = json!("changed");
            assert_eq!(
                load(reserved).err(),
                Some("invalid_configuration"),
                "{name}"
            );
        }

        let mut digest_changed = configuration(endpoint);
        digest_changed["expectedInstructionSha256"] = json!(sha("different"));
        assert_eq!(
            load(digest_changed).err(),
            Some("instruction_identity_mismatch")
        );

        let mut no_key = configuration(endpoint);
        no_key["apiKeyEnv"] = json!("MISSING_KEY");
        assert_eq!(load(no_key).err(), Some("model_key_unavailable"));

        let mut weights = configuration(endpoint);
        weights["weightsSha256"] = json!("not-a-digest");
        assert_eq!(load(weights).err(), Some("invalid_configuration"));

        let mut temperature = configuration(endpoint);
        temperature["temperature"] = json!(9.0);
        assert_eq!(load(temperature).err(), Some("invalid_configuration"));
    }
}
