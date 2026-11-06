//! The Codex run route, which starts `codex exec` on the operator's own account
//! in the server's working folder. It is off unless `LFG_CODEX_RUNS_TOKEN` is
//! set, and then serves only a loopback listener and loopback peers that send
//! that token with an expected Host and Origin. Every check runs before the
//! request body is parsed.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{ConnectInfo, DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    routing::post,
};

use crate::local_access::{self, LocalAccessError};
use crate::{AppState, error_response};

pub(crate) const PATH: &str = "/api/shippin/intelligence/lfg/runs";
const TOKEN_ENV: &str = "LFG_CODEX_RUNS_TOKEN";
const HOSTS_ENV: &str = "LFG_CODEX_RUNS_ALLOWED_HOSTS";
const ORIGINS_ENV: &str = "LFG_CODEX_RUNS_ALLOWED_ORIGINS";

#[derive(Clone)]
pub(crate) struct Api {
    state: AppState,
    listener: SocketAddr,
    token: Option<Arc<String>>,
    allowed_hosts: Arc<Vec<String>>,
    allowed_origins: Arc<Vec<String>>,
}

fn listed(name: &str, valid: fn(&str) -> bool) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty() && valid(value))
        .map(str::to_owned)
        .collect()
}

impl Api {
    fn from_environment(state: AppState, listener: SocketAddr) -> Self {
        Self::new(
            state,
            listener,
            std::env::var(TOKEN_ENV).ok(),
            listed(HOSTS_ENV, local_access::valid_authority),
            listed(ORIGINS_ENV, local_access::valid_origin),
        )
    }

    pub(crate) fn new(
        state: AppState,
        listener: SocketAddr,
        token: Option<String>,
        allowed_hosts: Vec<String>,
        allowed_origins: Vec<String>,
    ) -> Self {
        Self {
            state,
            listener,
            token: token
                .filter(|value| local_access::valid_token(value))
                .map(Arc::new),
            allowed_hosts: Arc::new(allowed_hosts),
            allowed_origins: Arc::new(allowed_origins),
        }
    }

    pub(crate) fn into_router(self) -> Router {
        let max_body = self.state.operations().max_http_body_bytes();
        Router::new()
            .route(PATH, post(run))
            .layer(DefaultBodyLimit::max(max_body))
            .with_state(self)
    }
}

pub(crate) fn routes(state: AppState, listener: SocketAddr) -> Router {
    Api::from_environment(state, listener).into_router()
}

async fn run(
    State(api): State<Api>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if api.token.is_none() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "codex_runs_disabled",
            "Codex runs are off. Set LFG_CODEX_RUNS_TOKEN on a loopback listener to enable them.",
        )
        .into_response();
    }
    if let Err(error) = local_access::authorize(
        api.listener,
        peer,
        &headers,
        api.token.as_deref().map(String::as_str),
        &api.allowed_hosts,
        &api.allowed_origins,
    ) {
        let (status, code) = match error {
            LocalAccessError::NonLoopback => (StatusCode::FORBIDDEN, "loopback_required"),
            LocalAccessError::InvalidHost => (StatusCode::FORBIDDEN, "invalid_host"),
            LocalAccessError::InvalidOrigin => (StatusCode::FORBIDDEN, "invalid_origin"),
            LocalAccessError::InvalidToken => (StatusCode::FORBIDDEN, "invalid_token"),
            LocalAccessError::TokenUnavailable => {
                (StatusCode::SERVICE_UNAVAILABLE, "codex_runs_disabled")
            }
        };
        return error_response(status, code, code).into_response();
    }
    let request = match Json::from_bytes(&body) {
        Ok(Json(request)) => request,
        Err(rejection) => return rejection.into_response(),
    };
    crate::lfg_run_codex(State(api.state), Json(request))
        .await
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, extract::connect_info::MockConnectInfo, http::Request};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef";

    async fn call(
        api: Api,
        peer: &str,
        headers: &[(&str, &str)],
        text: &str,
    ) -> (StatusCode, Value) {
        let app = api
            .into_router()
            .layer(MockConnectInfo(peer.parse::<SocketAddr>().unwrap()));
        let mut request = Request::builder()
            .method("POST")
            .uri(PATH)
            .header("content-type", "application/json");
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let body = json!({"sessionId": "s", "revision": 1, "text": text}).to_string();
        let response = app
            .oneshot(request.body(Body::from(body)).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    fn api(listener: &str, token: Option<&str>) -> Api {
        Api::new(
            AppState::new(),
            listener.parse().unwrap(),
            token.map(str::to_owned),
            Vec::new(),
            Vec::new(),
        )
    }

    #[tokio::test]
    async fn runs_are_off_without_a_token() {
        let (status, body) = call(api("127.0.0.1:4317", None), "127.0.0.1:5000", &[], "hi").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["code"], "codex_runs_disabled");
        // A weak token is treated as no token.
        let (status, _) = call(
            api("127.0.0.1:4317", Some("short")),
            "127.0.0.1:5000",
            &[],
            "hi",
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn only_a_loopback_caller_with_the_token_and_expected_host_reaches_codex() {
        let host = [("host", "127.0.0.1:4317"), ("x-lfg-session-token", TOKEN)];
        // A container or network listener never starts runs, even with the token.
        let (status, body) = call(
            api("0.0.0.0:4317", Some(TOKEN)),
            "127.0.0.1:5000",
            &host,
            "hi",
        )
        .await;
        assert_eq!(
            (status, body["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("loopback_required"))
        );
        let (status, body) = call(
            api("127.0.0.1:4317", Some(TOKEN)),
            "192.168.1.5:5000",
            &host,
            "hi",
        )
        .await;
        assert_eq!(
            (status, body["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("loopback_required"))
        );
        // DNS rebinding arrives with another Host; a cross-site page with another Origin.
        let rebinding = [
            ("host", "attacker.example:4317"),
            ("x-lfg-session-token", TOKEN),
        ];
        let (status, body) = call(
            api("127.0.0.1:4317", Some(TOKEN)),
            "127.0.0.1:5000",
            &rebinding,
            "hi",
        )
        .await;
        assert_eq!(
            (status, body["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("invalid_host"))
        );
        let cross = [
            ("host", "127.0.0.1:4317"),
            ("origin", "https://attacker.example"),
            ("x-lfg-session-token", TOKEN),
        ];
        let (status, body) = call(
            api("127.0.0.1:4317", Some(TOKEN)),
            "127.0.0.1:5000",
            &cross,
            "hi",
        )
        .await;
        assert_eq!(
            (status, body["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("invalid_origin"))
        );
        let wrong = [
            ("host", "127.0.0.1:4317"),
            (
                "x-lfg-session-token",
                "ffffffffffffffffffffffffffffffffffffffffffffffff",
            ),
        ];
        let (status, body) = call(
            api("127.0.0.1:4317", Some(TOKEN)),
            "127.0.0.1:5000",
            &wrong,
            "hi",
        )
        .await;
        assert_eq!(
            (status, body["code"].as_str()),
            (StatusCode::FORBIDDEN, Some("invalid_token"))
        );
        // With every check passed the request reaches the handler, whose own prompt
        // validation runs first; an empty prompt shows that without starting Codex.
        let (status, _) = call(
            api("127.0.0.1:4317", Some(TOKEN)),
            "127.0.0.1:5000",
            &host,
            "",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn the_base_router_no_longer_serves_runs() {
        let response = crate::router(AppState::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(PATH)
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"sessionId":"s","revision":1,"text":"hi"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
