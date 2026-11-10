//! Explicit, bounded local capture of prompt-free developer statistics.
use std::{fs::OpenOptions, io::Write, net::SocketAddr, path::PathBuf, time::Duration};

use bytes::Bytes;
use http_body_util::{BodyExt, Empty};
use hyper_util::{
    client::legacy::{Client, connect::HttpConnector},
    rt::TokioExecutor,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Preflight,
    Classification,
    Assessment,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Ok,
    Error,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Trace {
    sequence: u64,
    recorded_at: String,
    operation: Operation,
    outcome: Outcome,
    total_micros: u64,
    queue_micros: Option<u64>,
    work_micros: Option<u64>,
    // The running assessment prototype can attach configured rule details.
    // Accept that field but never persist its potentially identifying values.
    #[serde(default, skip_serializing, rename = "rules")]
    _rules: Option<serde::de::IgnoredAny>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Summary {
    operation: Operation,
    samples: usize,
    errors: usize,
    p50_total_micros: Option<u64>,
    p95_total_micros: Option<u64>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Snapshot {
    schema_version: String,
    enabled: bool,
    capacity: usize,
    total_recorded: u64,
    evicted: u64,
    cpu_time_micros: Option<u64>,
    operations_digest: String,
    encoder_digest: String,
    classifier_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    assessment_digest: Option<String>,
    summaries: Vec<Summary>,
    traces: Vec<Trace>,
}

impl Snapshot {
    fn parse(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let value: Self = serde_json::from_slice(bytes)?;
        let digest = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        };
        let timestamp = |s: &str| {
            !s.is_empty()
                && s.len() <= 40
                && s.bytes()
                    .all(|c| c.is_ascii_digit() || b"-T:.+Z".contains(&c))
        };
        if value.schema_version != "lfg.dev-stats.v1"
            || value.traces.len() > value.capacity
            || !digest(&value.operations_digest)
            || !digest(&value.encoder_digest)
            || !digest(&value.classifier_digest)
            || value
                .assessment_digest
                .as_deref()
                .is_some_and(|hash| !digest(hash))
            || value
                .traces
                .iter()
                .any(|trace| !timestamp(&trace.recorded_at))
        {
            return Err("invalid developer statistics snapshot".into());
        }
        Ok(value)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Entry {
    schema_version: &'static str,
    captured_at: String,
    snapshot: Snapshot,
}

async fn fetch(
    server: SocketAddr,
    timeout_ms: u64,
    max_response_bytes: usize,
) -> Result<Snapshot, Box<dyn std::error::Error>> {
    let client: Client<HttpConnector, Empty<Bytes>> =
        Client::builder(TokioExecutor::new()).build_http();
    let request = async {
        let uri = format!("http://{server}/api/lfg/dev-stats").parse()?;
        let mut response = client.get(uri).await?;
        if response.status() != hyper::StatusCode::OK {
            return Err("developer statistics request failed".into());
        }
        let mut bytes = Vec::new();
        while let Some(frame) = response.body_mut().frame().await {
            let frame = frame?;
            if let Some(chunk) = frame.data_ref() {
                if chunk.len() > max_response_bytes.saturating_sub(bytes.len()) {
                    return Err("developer statistics response exceeds configured limit".into());
                }
                bytes.extend_from_slice(chunk);
            }
        }
        Snapshot::parse(&bytes)
    };
    tokio::time::timeout(Duration::from_millis(timeout_ms), request).await?
}

pub async fn run(
    server: SocketAddr,
    output: PathBuf,
    interval_ms: u64,
    timeout_ms: u64,
    max_response_bytes: usize,
    max_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if !server.ip().is_loopback()
        || server.port() == 0
        || interval_ms == 0
        || timeout_ms == 0
        || max_response_bytes == 0
        || max_bytes == 0
    {
        return Err("loopback server and positive capture limits are required".into());
    }
    let first = fetch(server, timeout_ms, max_response_bytes).await?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut total = 0u64;
    let mut next = first;
    let mut previous: Option<Vec<u8>> = None;
    loop {
        let fingerprint = serde_json::to_vec(&next)?;
        if previous.as_deref() != Some(fingerprint.as_slice()) {
            let entry = Entry {
                schema_version: "lfg.dev-log-entry.v1",
                captured_at: lfg_core::now_rfc3339(),
                snapshot: next,
            };
            let mut line = serde_json::to_vec(&entry)?;
            line.push(b'\n');
            if u64::try_from(line.len())? > max_bytes.saturating_sub(total) {
                eprintln!("Capture stopped before exceeding the configured file limit.");
                return Ok(());
            }
            file.write_all(&line)?;
            file.flush()?;
            total += u64::try_from(line.len())?;
            previous = Some(fingerprint);
        }
        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        next = fetch(server, timeout_ms, max_response_bytes).await?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> serde_json::Value {
        serde_json::json!({"schemaVersion":"lfg.dev-stats.v1", "enabled":true,"capacity":1,
            "totalRecorded":1,"evicted":0,"cpuTimeMicros":null,"operationsDigest":"a".repeat(64),
            "encoderDigest":"b".repeat(64),"classifierDigest":"c".repeat(64),"summaries":[],
            "traces":[{"sequence":1,"recordedAt":"2026-11-01T12:00:00Z","operation":"preflight",
                "outcome":"ok","totalMicros":12,"queueMicros":1,"workMicros":9}]})
    }

    #[test]
    fn snapshot_rejects_prompt_fields_and_noncanonical_metadata() {
        assert!(Snapshot::parse(&serde_json::to_vec(&example()).unwrap()).is_ok());
        let mut value = example();
        value["traces"][0]["prompt"] = "private".into();
        assert!(Snapshot::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = example();
        value["traces"][0]["recordedAt"] = "private".into();
        assert!(Snapshot::parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn assessment_rule_details_are_accepted_but_not_written() {
        let mut value = example();
        value["assessmentDigest"] = "d".repeat(64).into();
        value["traces"][0]["operation"] = "assessment".into();
        value["traces"][0]["rules"] = serde_json::json!(["private-rule-detail"]);
        let snapshot = Snapshot::parse(&serde_json::to_vec(&value).unwrap()).unwrap();
        let serialized = serde_json::to_string(&snapshot).unwrap();
        assert!(!serialized.contains("private-rule-detail"));
        assert!(!serialized.contains("rules"));
        assert!(serialized.contains("assessmentDigest"));
    }

    #[tokio::test]
    async fn capture_is_bounded_and_never_overwrites_existing_files() {
        use axum::{Json, Router, routing::get};
        use std::sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicU64::new(0));
        let observed = requests.clone();
        let task = tokio::spawn(async move {
            let app = Router::new().route(
                "/api/lfg/dev-stats",
                get(move || {
                    let requests = requests.clone();
                    async move {
                        let count = requests.fetch_add(1, Ordering::Relaxed);
                        let mut value = example();
                        if count >= 3 {
                            value["totalRecorded"] = (count + 1).into();
                        }
                        Json(value)
                    }
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("diagnostics.jsonl");
        let entry = Entry {
            schema_version: "lfg.dev-log-entry.v1",
            captured_at: lfg_core::now_rfc3339(),
            snapshot: Snapshot::parse(&serde_json::to_vec(&example()).unwrap()).unwrap(),
        };
        let limit = u64::try_from(serde_json::to_vec(&entry).unwrap().len() + 32).unwrap();
        run(server, output.clone(), 1, 1_000, 10_000, limit)
            .await
            .unwrap();
        let contents = std::fs::read(&output).unwrap();
        assert!(contents.len() as u64 <= limit);
        assert_eq!(contents.iter().filter(|&&byte| byte == b'\n').count(), 1);
        assert!(observed.load(Ordering::Relaxed) >= 4);
        let saved: serde_json::Value = serde_json::from_slice(&contents).unwrap();
        assert_eq!(saved["snapshot"]["traces"][0]["totalMicros"], 12);
        assert!(
            run(server, output.clone(), 1, 1_000, 10_000, limit)
                .await
                .is_err()
        );
        assert_eq!(std::fs::read(output).unwrap(), contents);
        task.abort();
    }
}
