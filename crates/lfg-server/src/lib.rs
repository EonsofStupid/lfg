use std::sync::Arc;

use std::{convert::Infallible, time::Duration};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
};
use lfg_codex::{CodexRunner, CodexStatus, ExecutionResult};
use lfg_core::{
    CategoryClassifier, ClassificationSnapshot, ClassifierConfig, ClassifyRequest,
    CpuCategoryClassifier, ExperimentStatistics, GraphSnapshot, PreflightRequest,
    PreflightSnapshot, PromotionDecision, PromotionDecisionRequest, SemanticGrade,
    SemanticGradeRequest, SignalEnvelope,
};
use lfg_runtime::{
    CommitPlanRequest, ExecutionMetrics, LfgRuntime, PlanCommit, RuntimeError, RuntimeSignalStore,
    RuntimeStatus,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, broadcast};
use tokio_stream::{StreamExt, wrappers::BroadcastStream};

#[derive(Clone)]
pub struct AppState {
    lfg: Arc<Mutex<LfgRuntime<RuntimeSignalStore>>>,
    lfg_events: broadcast::Sender<SignalEnvelope>,
    codex: CodexRunner,
    classifier_config: Arc<ClassifierConfig>,
}

impl AppState {
    pub fn new() -> Self {
        let (lfg_events, _) = broadcast::channel(512);
        Self {
            lfg: Arc::new(Mutex::new(LfgRuntime::new(RuntimeSignalStore::memory()))),
            lfg_events,
            codex: CodexRunner::from_environment(
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            ),
            classifier_config: Arc::new(load_classifier_config()),
        }
    }

    pub fn persistent(signal_path: impl AsRef<std::path::Path>) -> Result<Self, RuntimeError> {
        let (lfg_events, _) = broadcast::channel(512);
        Ok(Self {
            lfg: Arc::new(Mutex::new(LfgRuntime::new(RuntimeSignalStore::jsonl(
                signal_path,
            )?))),
            lfg_events,
            codex: CodexRunner::from_environment(
                std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
            ),
            classifier_config: Arc::new(load_classifier_config()),
        })
    }
}

fn load_classifier_config() -> ClassifierConfig {
    let path = match std::env::var_os("LFG_CLASSIFIER_CONFIG") {
        Some(path) => path,
        None => return ClassifierConfig::default_alpha(),
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => match ClassifierConfig::from_json_str(&text) {
            Ok(config) => config,
            Err(error) => {
                eprintln!(
                    "LFG_CLASSIFIER_CONFIG at {path:?} failed to parse ({error}); using the default taxonomy"
                );
                ClassifierConfig::default_alpha()
            }
        },
        Err(error) => {
            eprintln!(
                "LFG_CLASSIFIER_CONFIG at {path:?} could not be read ({error}); using the default taxonomy"
            );
            ClassifierConfig::default_alpha()
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/shippin/intelligence/lfg/status", get(lfg_status))
        .route(
            "/api/shippin/intelligence/lfg/providers/codex",
            get(lfg_codex_status),
        )
        .route(
            "/api/shippin/intelligence/lfg/preflight",
            post(lfg_preflight),
        )
        .route("/api/shippin/intelligence/lfg/plans", post(lfg_commit_plan))
        .route("/api/shippin/intelligence/lfg/runs", post(lfg_run_codex))
        .route("/api/shippin/intelligence/lfg/signals", get(lfg_signals))
        .route(
            "/api/shippin/intelligence/lfg/signals/replay",
            get(lfg_replay),
        )
        .route(
            "/api/shippin/intelligence/lfg/workflows/{workflow_id}/graph",
            get(lfg_graph),
        )
        .route(
            "/api/shippin/intelligence/lfg/workflows/{workflow_id}/stream",
            get(lfg_stream),
        )
        .route(
            "/api/shippin/intelligence/lfg/evaluations/rubrics",
            get(lfg_evaluation_contracts),
        )
        .route(
            "/api/shippin/intelligence/lfg/evaluations/grades",
            post(lfg_record_grade),
        )
        .route(
            "/api/shippin/intelligence/lfg/evaluations/statistics",
            get(lfg_statistics),
        )
        .route(
            "/api/shippin/intelligence/lfg/evaluations/promotions",
            get(lfg_promotions).post(lfg_decide_promotion),
        )
        .route("/api/lfg/classify", post(lfg_classify))
        .with_state(state)
}

async fn lfg_classify(
    State(state): State<AppState>,
    Json(request): Json<ClassifyRequest>,
) -> Result<Json<ClassificationSnapshot>, (StatusCode, Json<ErrorResponse>)> {
    if request.session_id.trim().is_empty() || request.text.len() > 12_000 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "sessionId and up to 12,000 characters of text are required",
        ));
    }
    Ok(Json(
        CpuCategoryClassifier.classify(&request, &state.classifier_config),
    ))
}

pub fn router_with_ui(state: AppState, ui_directory: impl AsRef<std::path::Path>) -> Router {
    let ui_directory = ui_directory.as_ref().to_path_buf();
    router(state).fallback_service(
        tower_http::services::ServeDir::new(&ui_directory)
            .append_index_html_on_directories(true)
            .fallback(tower_http::services::ServeFile::new(
                ui_directory.join("index.html"),
            )),
    )
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "lfg-server",
    })
}

async fn lfg_status(State(state): State<AppState>) -> Json<RuntimeStatus> {
    Json(state.lfg.lock().await.status())
}

async fn lfg_codex_status(State(state): State<AppState>) -> Json<CodexStatus> {
    let runner = state.codex.clone();
    let status = tokio::task::spawn_blocking(move || runner.probe())
        .await
        .unwrap_or_else(|_| CodexStatus {
            target_id: "openai-codex-seat",
            adapter_id: lfg_codex::ADAPTER_ID,
            adapter_version: lfg_codex::ADAPTER_VERSION,
            provider: "openai",
            display_name: "OpenAI Codex subscription",
            status: "unavailable".into(),
            auth_class: "subscription-seat",
            auth_method: "unknown".into(),
            cli_version: None,
            workspace: "workspace".into(),
            detail: "Codex readiness probe could not complete".into(),
            capabilities: vec![],
        });
    state
        .lfg
        .lock()
        .await
        .set_codex_availability(status.active());
    Json(status)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CodexRunRequest {
    session_id: String,
    revision: u64,
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CodexRunResponse {
    workflow_id: String,
    graph: GraphSnapshot,
    result: ExecutionResult,
}

async fn lfg_run_codex(
    State(state): State<AppState>,
    Json(request): Json<CodexRunRequest>,
) -> Result<(StatusCode, Json<CodexRunResponse>), (StatusCode, Json<ErrorResponse>)> {
    let runner = state.codex.clone();
    let probe_runner = runner.clone();
    let provider = tokio::task::spawn_blocking(move || probe_runner.probe())
        .await
        .map_err(|_| {
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "provider_probe_failed",
                "Codex readiness probe could not complete",
            )
        })?;
    if !provider.active() {
        return Err(error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "provider_inactive",
            &provider.detail,
        ));
    }
    let (commit, binding) = {
        let mut runtime = state.lfg.lock().await;
        runtime.set_codex_availability(true);
        let commit = runtime
            .commit_plan(CommitPlanRequest {
                session_id: request.session_id,
                revision: request.revision,
                text: request.text.clone(),
                include_fixture_bindings: false,
            })
            .map_err(lfg_error)?;
        let binding = runtime
            .commit_codex_binding(&commit.workflow_id, &commit.plan)
            .map_err(lfg_error)?;
        runtime
            .record_execution_started(&commit.workflow_id, &binding)
            .map_err(lfg_error)?;
        (commit, binding)
    };
    let prompt = request.text;
    let execution = tokio::task::spawn_blocking(move || runner.execute(&prompt))
        .await
        .map_err(|_| {
            error_response(
                StatusCode::BAD_GATEWAY,
                "provider_execution_failed",
                "Codex execution task could not complete",
            )
        })?;
    let mut runtime = state.lfg.lock().await;
    match execution {
        Ok(result) => {
            for event in &result.events {
                runtime
                    .record_execution_event(
                        &commit.workflow_id,
                        &binding,
                        &event.event_type,
                        &event.operation,
                        &event.status,
                        event.native_payload_length,
                    )
                    .map_err(lfg_error)?;
            }
            runtime
                .record_execution_completed(
                    &commit.workflow_id,
                    &binding,
                    &ExecutionMetrics {
                        input_tokens: result.usage.input_tokens,
                        cached_input_tokens: result.usage.cached_input_tokens,
                        output_tokens: result.usage.output_tokens,
                        reasoning_output_tokens: result.usage.reasoning_output_tokens,
                        response_length: result.final_response.len(),
                    },
                )
                .map_err(lfg_error)?;
            let graph = runtime
                .graph(&commit.workflow_id, None)
                .map_err(lfg_error)?;
            let signals = runtime
                .signals(Some(&commit.workflow_id))
                .map_err(lfg_error)?;
            drop(runtime);
            for signal in signals {
                let _ = state.lfg_events.send(signal);
            }
            Ok((
                StatusCode::CREATED,
                Json(CodexRunResponse {
                    workflow_id: commit.workflow_id,
                    graph,
                    result,
                }),
            ))
        }
        Err(error) => {
            runtime
                .record_execution_failed(&commit.workflow_id, &binding, "provider_execution")
                .map_err(lfg_error)?;
            Err(error_response(
                StatusCode::BAD_GATEWAY,
                "provider_execution_failed",
                &error.to_string(),
            ))
        }
    }
}

async fn lfg_preflight(
    State(state): State<AppState>,
    Json(request): Json<PreflightRequest>,
) -> Result<Json<PreflightSnapshot>, (StatusCode, Json<ErrorResponse>)> {
    state
        .lfg
        .lock()
        .await
        .preflight(&request)
        .map(Json)
        .map_err(lfg_error)
}

async fn lfg_commit_plan(
    State(state): State<AppState>,
    Json(request): Json<CommitPlanRequest>,
) -> Result<(StatusCode, Json<PlanCommit>), (StatusCode, Json<ErrorResponse>)> {
    let mut runtime = state.lfg.lock().await;
    let result = runtime.commit_plan(request).map_err(lfg_error)?;
    let signals = runtime
        .signals(Some(&result.workflow_id))
        .map_err(lfg_error)?;
    drop(runtime);
    for signal in signals {
        let _ = state.lfg_events.send(signal);
    }
    Ok((StatusCode::CREATED, Json(result)))
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignalsQuery {
    workflow_id: Option<String>,
    after_sequence: Option<u64>,
    through_sequence: Option<u64>,
    limit: Option<usize>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SignalsResponse {
    signals: Vec<SignalEnvelope>,
    next_sequence: u64,
}

async fn lfg_signals(
    State(state): State<AppState>,
    Query(query): Query<SignalsQuery>,
) -> Result<Json<SignalsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let runtime = state.lfg.lock().await;
    let mut signals = runtime
        .signals(query.workflow_id.as_deref())
        .map_err(lfg_error)?;
    if let Some(after) = query.after_sequence {
        signals.retain(|signal| signal.data.sequence > after);
    }
    if let Some(through) = query.through_sequence {
        signals.retain(|signal| signal.data.sequence <= through);
    }
    signals.truncate(query.limit.unwrap_or(100).clamp(1, 500));
    let next_sequence = signals
        .last()
        .map(|signal| signal.data.sequence)
        .unwrap_or(query.after_sequence.unwrap_or(0));
    Ok(Json(SignalsResponse {
        signals,
        next_sequence,
    }))
}

async fn lfg_replay(
    State(state): State<AppState>,
    Query(query): Query<SignalsQuery>,
) -> Result<Json<SignalsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let workflow_id = query
        .workflow_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "workflowId is required",
            )
        })?;
    let signals = state
        .lfg
        .lock()
        .await
        .replay(
            workflow_id,
            query.after_sequence.unwrap_or(0),
            query.limit.unwrap_or(100),
        )
        .map_err(lfg_error)?;
    let next_sequence = signals
        .last()
        .map(|signal| signal.data.sequence)
        .unwrap_or(query.after_sequence.unwrap_or(0));
    Ok(Json(SignalsResponse {
        signals,
        next_sequence,
    }))
}

async fn lfg_graph(
    State(state): State<AppState>,
    Path(workflow_id): Path<String>,
    Query(query): Query<SignalsQuery>,
) -> Result<Json<GraphSnapshot>, (StatusCode, Json<ErrorResponse>)> {
    state
        .lfg
        .lock()
        .await
        .graph(&workflow_id, query.through_sequence)
        .map(Json)
        .map_err(lfg_error)
}

async fn lfg_stream(
    State(state): State<AppState>,
    Path(workflow_id): Path<String>,
    headers: HeaderMap,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let after_sequence = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let stream =
        BroadcastStream::new(state.lfg_events.subscribe()).filter_map(move |item| match item {
            Ok(signal)
                if signal.data.workflow_id == workflow_id
                    && signal.data.sequence > after_sequence =>
            {
                let data = serde_json::to_string(&signal).unwrap_or_else(|_| "{}".into());
                Some(Ok(Event::default()
                    .id(signal.data.sequence.to_string())
                    .event("lfg.signal")
                    .data(data)))
            }
            _ => None,
        });
    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn lfg_evaluation_contracts(State(state): State<AppState>) -> Json<Value> {
    let runtime = state.lfg.lock().await;
    Json(
        serde_json::json!({"rubrics": [runtime.rubric()], "promotionPolicy": runtime.promotion_policy()}),
    )
}

async fn lfg_record_grade(
    State(state): State<AppState>,
    Json(request): Json<SemanticGradeRequest>,
) -> Result<(StatusCode, Json<SemanticGrade>), (StatusCode, Json<ErrorResponse>)> {
    state
        .lfg
        .lock()
        .await
        .record_grade(request)
        .map(|grade| (StatusCode::CREATED, Json(grade)))
        .map_err(lfg_error)
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EvaluationQuery {
    experiment_id: Option<String>,
}

async fn lfg_statistics(
    State(state): State<AppState>,
    Query(query): Query<EvaluationQuery>,
) -> Result<Json<ExperimentStatistics>, (StatusCode, Json<ErrorResponse>)> {
    let experiment_id = query
        .experiment_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "experimentId is required",
            )
        })?;
    state
        .lfg
        .lock()
        .await
        .statistics(experiment_id)
        .map(Json)
        .map_err(lfg_error)
}

async fn lfg_decide_promotion(
    State(state): State<AppState>,
    Json(request): Json<PromotionDecisionRequest>,
) -> Result<(StatusCode, Json<PromotionDecision>), (StatusCode, Json<ErrorResponse>)> {
    state
        .lfg
        .lock()
        .await
        .decide_promotion(request)
        .map(|decision| (StatusCode::CREATED, Json(decision)))
        .map_err(lfg_error)
}

async fn lfg_promotions(
    State(state): State<AppState>,
    Query(query): Query<EvaluationQuery>,
) -> Result<Json<Value>, (StatusCode, Json<ErrorResponse>)> {
    state
        .lfg
        .lock()
        .await
        .promotion_decisions(query.experiment_id.as_deref())
        .map(|decisions| Json(serde_json::json!({"decisions": decisions})))
        .map_err(lfg_error)
}

#[derive(Clone, Serialize)]
struct ErrorResponse {
    code: &'static str,
    message: String,
}

fn lfg_error(error: RuntimeError) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        RuntimeError::InvalidRequest(message) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                code: "invalid_request",
                message,
            }),
        ),
        RuntimeError::GradeExists => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                code: "grade_exists",
                message: error.to_string(),
            }),
        ),
        RuntimeError::PromotionGate => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                code: "promotion_gate_not_satisfied",
                message: error.to_string(),
            }),
        ),
        RuntimeError::WorkflowNotFound(_) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                code: "workflow_not_found",
                message: error.to_string(),
            }),
        ),
        RuntimeError::Protocol(_) | RuntimeError::SignalConflict(_) => (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                code: "routing_contract_rejected",
                message: error.to_string(),
            }),
        ),
        RuntimeError::Storage(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                code: "event_store_unavailable",
                message: "The LFG event store is unavailable".into(),
            }),
        ),
    }
}

fn error_response(
    status: StatusCode,
    code: &'static str,
    message: &str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            code,
            message: message.into(),
        }),
    )
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

    async fn json_response(app: Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    #[tokio::test]
    async fn classify_reports_category_percentages_and_escalation() {
        let app = router(AppState::new());
        let request = Request::builder()
            .method("POST")
            .uri("/api/lfg/classify")
            .header("content-type", "application/json")
            .body(Body::from(json!({"sessionId":"classify-test","revision":1,"text":"First, refactor the auth module across multiple files, then implement a new session store, and finally migrate the tests step by step.","activeSubscriptions":["claude"],"routingNeed":"balanced"}).to_string()))
            .unwrap();
        let (status, snapshot) = json_response(app, request).await;
        assert_eq!(status, StatusCode::OK);
        let categories = snapshot["categories"].as_array().unwrap();
        assert!(categories.len() >= 7);
        assert_eq!(snapshot["needsBiggerModel"], true);
        assert!(!snapshot["escalationCatalog"].as_array().unwrap().is_empty());
        assert_eq!(snapshot["recommendedEscalation"]["model"], "sonnet-5");
    }

    #[tokio::test]
    async fn classify_with_no_active_subscriptions_still_returns_the_full_catalog_inactive() {
        let app = router(AppState::new());
        let request = Request::builder()
            .method("POST")
            .uri("/api/lfg/classify")
            .header("content-type", "application/json")
            .body(Body::from(json!({"sessionId":"classify-test-2","revision":1,"text":"First, refactor the auth module across multiple files, then implement a new session store, and finally migrate the tests step by step."}).to_string()))
            .unwrap();
        let (status, snapshot) = json_response(app, request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(snapshot["recommendedEscalation"].is_null());
        let catalog = snapshot["escalationCatalog"].as_array().unwrap();
        assert!(!catalog.is_empty());
        assert!(catalog.iter().all(|candidate| candidate["active"] == false));
    }

    #[tokio::test]
    async fn preflight_and_committed_graph_use_real_derived_tallies() {
        let app = router(AppState::new());
        let preflight_request = Request::builder()
            .method("POST")
            .uri("/api/shippin/intelligence/lfg/preflight")
            .header("content-type", "application/json")
            .body(Body::from(json!({"sessionId":"api-test","revision":1,"text":"Research, implement, test, and grade an image workflow"}).to_string()))
            .unwrap();
        let (status, preflight) = json_response(app.clone(), preflight_request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(preflight["predictedAgents"].as_u64().unwrap() >= 4);
        assert!(preflight["durationMicros"].is_u64());

        let commit_request = Request::builder()
            .method("POST")
            .uri("/api/shippin/intelligence/lfg/plans")
            .header("content-type", "application/json")
            .body(Body::from(json!({"sessionId":"api-test","revision":1,"text":"Research, implement, test, and grade an image workflow","includeFixtureBindings":true}).to_string()))
            .unwrap();
        let (status, commit) = json_response(app.clone(), commit_request).await;
        assert_eq!(status, StatusCode::CREATED);
        let workflow_id = commit["workflowId"].as_str().unwrap();
        assert!(commit["graph"]["totals"]["committed"].as_u64().unwrap() >= 4);
        assert_eq!(commit["graph"]["totals"]["predicted"], 0);

        let graph_request = Request::builder()
            .uri(format!(
                "/api/shippin/intelligence/lfg/workflows/{workflow_id}/graph"
            ))
            .body(Body::empty())
            .unwrap();
        let (status, graph) = json_response(app, graph_request).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(graph, commit["graph"]);
    }

    #[tokio::test]
    async fn preflight_prompt_is_not_written_into_signals() {
        let app = router(AppState::new());
        let secret_marker = "private-user-prompt-marker-42";
        let request = Request::builder()
            .method("POST")
            .uri("/api/shippin/intelligence/lfg/plans")
            .header("content-type", "application/json")
            .body(Body::from(json!({"sessionId":"privacy-test","revision":1,"text":format!("implement and test {secret_marker}"),"includeFixtureBindings":false}).to_string()))
            .unwrap();
        let (status, commit) = json_response(app.clone(), request).await;
        assert_eq!(status, StatusCode::CREATED);
        let workflow_id = commit["workflowId"].as_str().unwrap();
        let signals_request = Request::builder()
            .uri(format!(
                "/api/shippin/intelligence/lfg/signals?workflowId={workflow_id}"
            ))
            .body(Body::empty())
            .unwrap();
        let (status, signals) = json_response(app, signals_request).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!signals.to_string().contains(secret_marker));
    }

    #[tokio::test]
    async fn status_is_truthful_before_probe_and_promotion_gate_is_enforced() {
        let app = router(AppState::new());
        let (status, runtime) = json_response(
            app.clone(),
            Request::builder()
                .uri("/api/shippin/intelligence/lfg/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(runtime["liveProviders"], 0);
        assert_eq!(runtime["configuredProviders"], 1);
        assert_eq!(runtime["fixtureProviders"], 4);

        let request = Request::builder().method("POST").uri("/api/shippin/intelligence/lfg/evaluations/promotions").header("content-type", "application/json").body(Body::from(json!({
            "experimentId":"empty-experiment", "baselineVariantId":"control", "candidateVariantId":"candidate", "action":"promote",
            "decidedBy":{"id":"operator","version":"1","sha256":"sha256:test"}, "reasonCodes":["manual-request"]
        }).to_string())).unwrap();
        let (status, error) = json_response(app, request).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(error["code"], "promotion_gate_not_satisfied");
    }
}
