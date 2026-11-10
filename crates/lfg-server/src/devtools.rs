//! Private, read-only scorecard results for the local devtools page.
//!
//! The panel file can hold redacted prompts and realized outcomes, so it is served only on a
//! loopback listener, to loopback peers, with the shared Host, Origin and session-token checks.
//! It is read on each request, so a new scorecard run appears without restarting the server.
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::UNIX_EPOCH};

use axum::{
    Json, Router,
    extract::{ConnectInfo, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};

const PANEL_SCHEMA: &str = "lfg.devtools-panel.v1";
const MAX_PANEL_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct DevtoolsApi {
    listener: SocketAddr,
    token: Option<Arc<String>>,
    panel: Option<Arc<PathBuf>>,
    hosts: Arc<Vec<String>>,
    origins: Arc<Vec<String>>,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            [(header::CACHE_CONTROL, "no-store")],
            Json(json!({"error":{"code":self.code,"message":self.message}})),
        )
            .into_response()
    }
}

fn list_env(name: &str, valid: fn(&str) -> bool) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| valid(value))
        .map(str::to_owned)
        .take(16)
        .collect()
}

impl DevtoolsApi {
    pub(crate) fn for_listener(listener: SocketAddr) -> Self {
        Self::new(
            listener,
            std::env::var("LFG_DEVTOOLS_TOKEN").ok(),
            std::env::var_os("LFG_DEVTOOLS_PANEL_FILE").map(PathBuf::from),
            list_env(
                "LFG_DEVTOOLS_ALLOWED_HOSTS",
                super::local_access::valid_authority,
            ),
            list_env(
                "LFG_DEVTOOLS_ALLOWED_ORIGINS",
                super::local_access::valid_origin,
            ),
        )
    }

    fn new(
        listener: SocketAddr,
        token: Option<String>,
        panel: Option<PathBuf>,
        hosts: Vec<String>,
        origins: Vec<String>,
    ) -> Self {
        Self {
            listener,
            token: token
                .filter(|value| super::local_access::valid_token(value))
                .map(Arc::new),
            panel: panel.map(Arc::new),
            hosts: Arc::new(hosts),
            origins: Arc::new(origins),
        }
    }

    pub(crate) fn routes(self) -> Router {
        Router::new()
            .route("/api/lfg/devtools/status", get(status))
            .route("/api/lfg/devtools/panel", get(panel))
            .with_state(self)
    }

    fn authorize(&self, peer: SocketAddr, headers: &HeaderMap) -> Result<(), ApiError> {
        use super::local_access::LocalAccessError as Denied;
        let refused = |status, code, message: &str| ApiError {
            status,
            code,
            message: message.into(),
        };
        super::local_access::authorize(
            self.listener,
            peer,
            headers,
            self.token.as_deref().map(String::as_str),
            &self.hosts,
            &self.origins,
        )
        .map_err(|denied| match denied {
            Denied::NonLoopback => refused(
                StatusCode::FORBIDDEN,
                "local_only",
                "devtools results are available only to loopback clients",
            ),
            Denied::InvalidHost => refused(
                StatusCode::FORBIDDEN,
                "invalid_host",
                "Host must match the configured loopback listener",
            ),
            Denied::InvalidOrigin => refused(
                StatusCode::FORBIDDEN,
                "invalid_origin",
                "Origin is not allowed for local devtools",
            ),
            Denied::TokenUnavailable => refused(
                StatusCode::SERVICE_UNAVAILABLE,
                "devtools_unavailable",
                "devtools results require LFG_DEVTOOLS_TOKEN with at least 32 non-whitespace bytes",
            ),
            Denied::InvalidToken => refused(
                StatusCode::FORBIDDEN,
                "invalid_session_token",
                "a valid local devtools session token is required",
            ),
        })
    }

    fn panel_path(&self) -> Result<&PathBuf, ApiError> {
        self.panel.as_deref().ok_or_else(|| ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: "devtools_unavailable",
            message: "no scorecard results are connected; configure LFG_DEVTOOLS_PANEL_FILE".into(),
        })
    }
}

fn unreadable(message: &str) -> ApiError {
    ApiError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        code: "panel_unavailable",
        message: message.into(),
    }
}

async fn status(
    State(api): State<DevtoolsApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Result<([(header::HeaderName, &'static str); 1], Json<Value>), ApiError> {
    api.authorize(peer, &headers)?;
    let metadata = tokio::fs::metadata(api.panel_path()?)
        .await
        .map_err(|_| unreadable("the scorecard results file cannot be read"))?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|elapsed| elapsed.as_millis() as u64);
    Ok((
        [(header::CACHE_CONTROL, "no-store")],
        Json(
            json!({"schemaVersion":"lfg.devtools-status.v1","modifiedMs":modified,"bytes":metadata.len()}),
        ),
    ))
}

async fn panel(
    State(api): State<DevtoolsApi>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Result<([(header::HeaderName, &'static str); 1], Json<Value>), ApiError> {
    api.authorize(peer, &headers)?;
    let path = api.panel_path()?;
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(|_| unreadable("the scorecard results file cannot be read"))?;
    if metadata.len() > MAX_PANEL_BYTES {
        return Err(unreadable("the scorecard results file is too large"));
    }
    let raw = tokio::fs::read(path)
        .await
        .map_err(|_| unreadable("the scorecard results file cannot be read"))?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|_| unreadable("the scorecard results file is not valid JSON"))?;
    if value.get("schemaVersion").and_then(Value::as_str) != Some(PANEL_SCHEMA) {
        return Err(unreadable(
            "the scorecard results file has an unsupported schema",
        ));
    }
    Ok(([(header::CACHE_CONTROL, "no-store")], Json(value)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    const TOKEN: &str = "abcdefghijklmnopqrstuvwxyz0123456789";

    fn listener() -> SocketAddr {
        "127.0.0.1:4317".parse().unwrap()
    }

    fn api(panel: Option<PathBuf>) -> DevtoolsApi {
        DevtoolsApi::new(listener(), Some(TOKEN.into()), panel, vec![], vec![])
    }

    async fn get_from(
        api: DevtoolsApi,
        path: &str,
        peer: &str,
        token: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut request = Request::builder()
            .uri(path)
            .header(header::HOST, listener().to_string());
        if let Some(token) = token {
            request = request.header("x-lfg-session-token", token);
        }
        let mut request = request.body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(peer.parse::<SocketAddr>().unwrap()));
        let response = api.routes().oneshot(request).await.unwrap();
        let status = response.status();
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    fn write_panel(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("panel.json");
        std::fs::write(&path, contents).unwrap();
        (directory, path)
    }

    #[tokio::test]
    async fn serves_the_current_file_to_a_local_client_with_the_token() {
        let (_directory, path) =
            write_panel(r#"{"schemaVersion":"lfg.devtools-panel.v1","rows":[1]}"#);
        let (status, body) = get_from(
            api(Some(path.clone())),
            "/api/lfg/devtools/panel",
            "127.0.0.1:50000",
            Some(TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["rows"][0], 1);

        std::fs::write(
            &path,
            r#"{"schemaVersion":"lfg.devtools-panel.v1","rows":[2]}"#,
        )
        .unwrap();
        let (_, body) = get_from(
            api(Some(path)),
            "/api/lfg/devtools/panel",
            "127.0.0.1:50000",
            Some(TOKEN),
        )
        .await;
        assert_eq!(body["rows"][0], 2);
    }

    #[tokio::test]
    async fn refuses_a_remote_peer_a_wrong_token_and_a_missing_token() {
        let (_directory, path) = write_panel(r#"{"schemaVersion":"lfg.devtools-panel.v1"}"#);
        for (peer, token, code) in [
            ("192.168.1.9:50000", Some(TOKEN), "local_only"),
            ("127.0.0.1:50000", Some("wrong"), "invalid_session_token"),
            ("127.0.0.1:50000", None, "invalid_session_token"),
        ] {
            let (status, body) = get_from(
                api(Some(path.clone())),
                "/api/lfg/devtools/panel",
                peer,
                token,
            )
            .await;
            assert_eq!(status, StatusCode::FORBIDDEN);
            assert_eq!(body["error"]["code"], code);
        }
    }

    #[tokio::test]
    async fn a_non_loopback_listener_never_serves_results() {
        let (_directory, path) = write_panel(r#"{"schemaVersion":"lfg.devtools-panel.v1"}"#);
        let exposed = DevtoolsApi::new(
            "0.0.0.0:4317".parse().unwrap(),
            Some(TOKEN.into()),
            Some(path),
            vec![],
            vec![],
        );
        let (status, body) = get_from(
            exposed,
            "/api/lfg/devtools/panel",
            "127.0.0.1:50000",
            Some(TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(body["error"]["code"], "local_only");
    }

    #[tokio::test]
    async fn explains_missing_configuration_and_rejects_unknown_files() {
        let without_file = get_from(
            api(None),
            "/api/lfg/devtools/status",
            "127.0.0.1:50000",
            Some(TOKEN),
        )
        .await;
        assert_eq!(without_file.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(without_file.1["error"]["code"], "devtools_unavailable");

        let without_token =
            DevtoolsApi::new(listener(), Some("short".into()), None, vec![], vec![]);
        let (status, body) = get_from(
            without_token,
            "/api/lfg/devtools/panel",
            "127.0.0.1:50000",
            Some("short"),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["error"]["code"], "devtools_unavailable");

        for contents in ["not json", r#"{"schemaVersion":"something.else"}"#] {
            let (_directory, path) = write_panel(contents);
            let (status, body) = get_from(
                api(Some(path)),
                "/api/lfg/devtools/panel",
                "127.0.0.1:50000",
                Some(TOKEN),
            )
            .await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["code"], "panel_unavailable");
        }
    }

    #[tokio::test]
    async fn status_reports_only_size_and_time() {
        let (_directory, path) = write_panel(r#"{"schemaVersion":"lfg.devtools-panel.v1"}"#);
        let (status, body) = get_from(
            api(Some(path)),
            "/api/lfg/devtools/status",
            "127.0.0.1:50000",
            Some(TOKEN),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["modifiedMs"].is_u64());
        assert!(body["bytes"].is_u64());
        assert_eq!(body.as_object().unwrap().len(), 3);
    }
}
