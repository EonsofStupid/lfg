use std::{collections::HashSet, net::SocketAddr, path::Path, time::Duration};

use bytes::Bytes;
use http_body_util::{BodyExt, Empty};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Config {
    schema_version: String,
    revision: String,
    endpoint: SocketAddr,
    timeout_ms: u64,
    poll_interval_ms: u64,
    max_response_bytes: usize,
}

#[derive(Clone)]
pub(crate) struct Source {
    config: Config,
    config_digest: String,
    client: Client<HttpConnector, Empty<Bytes>>,
}

#[derive(Deserialize)]
struct SeamState {
    registry: Registry,
}

#[derive(Deserialize)]
struct Registry {
    providers: Vec<Provider>,
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct Provider {
    id: String,
    activated: bool,
    capabilities: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Context {
    schema_version: &'static str,
    status: &'static str,
    revision: Option<String>,
    config_digest: Option<String>,
    poll_interval_ms: Option<u64>,
    providers: Option<Vec<Provider>>,
}

impl Source {
    fn parse(text: &str) -> Result<Self, String> {
        let config: Config = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if config.schema_version != "lfg.devpulse-context.v1"
            || config.revision.trim().is_empty()
            || !config.endpoint.ip().is_loopback()
            || config.endpoint.port() == 0
            || config.timeout_ms == 0
            || config.poll_interval_ms == 0
            || config.poll_interval_ms > i32::MAX as u64
            || config.max_response_bytes == 0
        {
            return Err("invalid DevPulse context configuration".into());
        }
        let client = Client::builder(TokioExecutor::new()).build_http();
        let config_digest = lfg_core::digest_value(
            &serde_json::to_value(&config).map_err(|error| error.to_string())?,
        );
        Ok(Self {
            config,
            config_digest,
            client,
        })
    }

    pub(crate) fn from_path(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {path:?}: {error}"))?;
        Self::parse(&text).map_err(|error| format!("invalid {path:?}: {error}"))
    }

    async fn providers(&self) -> Result<Vec<Provider>, ()> {
        let request = async {
            let uri = format!("http://{}/state", self.config.endpoint)
                .parse()
                .map_err(|_| ())?;
            let mut response = self.client.get(uri).await.map_err(|_| ())?;
            if response.status() != hyper::StatusCode::OK {
                return Err(());
            }
            let mut bytes = Vec::new();
            while let Some(frame) = response.body_mut().frame().await {
                let frame = frame.map_err(|_| ())?;
                if let Some(chunk) = frame.data_ref() {
                    if chunk.len() > self.config.max_response_bytes.saturating_sub(bytes.len()) {
                        return Err(());
                    }
                    bytes.extend_from_slice(chunk);
                }
            }
            Ok(bytes)
        };
        let bytes = tokio::time::timeout(Duration::from_millis(self.config.timeout_ms), request)
            .await
            .map_err(|_| ())??;
        let state: SeamState = serde_json::from_slice(&bytes).map_err(|_| ())?;
        let mut ids = HashSet::new();
        if state.registry.providers.iter().any(|p| {
            p.id.trim().is_empty()
                || !ids.insert(&p.id)
                || p.capabilities.iter().any(|c| c.trim().is_empty())
        }) {
            return Err(());
        }
        Ok(state.registry.providers)
    }

    pub(crate) async fn context(&self, template: &crate::intake::Template) -> Context {
        let providers = self.providers().await.ok().filter(|providers| {
            serde_json::to_value(providers).is_ok_and(|value| template.safe(&value))
        });
        Context {
            schema_version: "lfg.app-context.v1",
            status: if providers.is_some() {
                "ready"
            } else {
                "unavailable"
            },
            revision: Some(self.config.revision.clone()),
            config_digest: Some(self.config_digest.clone()),
            poll_interval_ms: Some(self.config.poll_interval_ms),
            providers,
        }
    }
}

pub(crate) fn disabled() -> Context {
    Context {
        schema_version: "lfg.app-context.v1",
        status: "disabled",
        revision: None,
        config_digest: None,
        poll_interval_ms: None,
        providers: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, routing::get};
    use serde_json::json;

    fn config(endpoint: SocketAddr, limit: usize) -> String {
        json!({"schemaVersion":"lfg.devpulse-context.v1","revision":"test", "endpoint":endpoint,
               "timeoutMs":1000,"pollIntervalMs":50,"maxResponseBytes":limit})
        .to_string()
    }

    #[test]
    fn rejects_nonlocal_and_invalid_configurations() {
        assert!(Source::parse(&config("192.0.2.1:7341".parse().unwrap(), 1000)).is_err());
        assert!(Source::parse(&config("127.0.0.1:7341".parse().unwrap(), 0)).is_err());
    }

    #[tokio::test]
    async fn projects_only_roster_facts_and_refuses_oversize() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/state",
            get(|| async {
                Json(
                    json!({"registry":{"providers":[{"id":"example", "activated":true,
                "capabilities":["read-sessions"], "note":"local history only", "label":"Example"}]},
                "other":"not exposed"}),
                )
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let template = crate::intake::Template::bundled();
        let context = Source::parse(&config(address, 4096))
            .unwrap()
            .context(&template)
            .await;
        let value = serde_json::to_value(context).unwrap();
        assert_eq!(value["status"], "ready");
        assert_eq!(value["configDigest"].as_str().unwrap().len(), 64);
        assert_eq!(value["providers"][0]["activated"], true);
        assert_eq!(value["providers"][0]["capabilities"][0], "read-sessions");
        assert!(value["providers"][0].get("note").is_none());
        let limited = Source::parse(&config(address, 8))
            .unwrap()
            .context(&template)
            .await;
        assert_eq!(limited.status, "unavailable");
        assert!(limited.providers.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn refuses_credential_shaped_provider_data() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/state",
            get(|| async {
                Json(
                    json!({"registry":{"providers":[{"id":"sk-secret-value", "activated":true,
                "capabilities":["read-sessions"]}]}}),
                )
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let result = Source::parse(&config(address, 4096))
            .unwrap()
            .context(&crate::intake::Template::bundled())
            .await;
        assert_eq!(result.status, "unavailable");
        assert!(result.providers.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn does_not_follow_redirects_and_bounds_slow_responses() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let reached = Arc::new(AtomicBool::new(false));
        let flag = reached.clone();
        let app = Router::new()
            .route(
                "/state",
                get(|| async { (axum::http::StatusCode::FOUND, [("location", "/target")]) }),
            )
            .route(
                "/target",
                get(move || {
                    let flag = flag.clone();
                    async move {
                        flag.store(true, Ordering::SeqCst);
                        Json(json!({"registry":{"providers":[]}}))
                    }
                }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let template = crate::intake::Template::bundled();
        assert_eq!(
            Source::parse(&config(address, 4096))
                .unwrap()
                .context(&template)
                .await
                .status,
            "unavailable"
        );
        assert!(!reached.load(Ordering::SeqCst));
        server.abort();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/state",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(200)).await;
                Json(json!({"registry":{"providers":[]}}))
            }),
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut value: serde_json::Value = serde_json::from_str(&config(address, 4096)).unwrap();
        value["timeoutMs"] = json!(20);
        assert_eq!(
            Source::parse(&value.to_string())
                .unwrap()
                .context(&template)
                .await
                .status,
            "unavailable"
        );
        server.abort();
    }
}
