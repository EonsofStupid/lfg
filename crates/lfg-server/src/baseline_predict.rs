use std::collections::HashSet;
use std::io::{self, BufRead, Read, Write};

use lfg_core::{PreflightEncoder, PreflightRequest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Example {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Prediction {
    schema_version: &'static str,
    id: String,
    input_sha256: String,
    producer_sha256: String,
    model: lfg_core::ManifestRef,
    intents: Vec<String>,
    capabilities: Vec<String>,
    risks: Vec<String>,
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let operations = lfg_server::configured_operations()?;
    let encoder = lfg_server::configured_preflight_encoder()?;
    let mut executable = std::fs::File::open(std::env::current_exe()?)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let read = executable.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let producer_sha256 = format!("{:x}", hasher.finalize());
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut ids = HashSet::new();
    let mut line = String::new();
    let mut count = 0_u64;
    loop {
        line.clear();
        let limit = u64::try_from(operations.max_http_body_bytes())?.saturating_add(1);
        let read = reader.by_ref().take(limit).read_line(&mut line)?;
        if read == 0 {
            break;
        }
        if read > operations.max_http_body_bytes() {
            return Err("input line exceeds the configured HTTP body byte limit".into());
        }
        if line.trim().is_empty() {
            continue;
        }
        let row: Example = serde_json::from_str(&line)?;
        if row.id.trim().is_empty() || !ids.insert(row.id.clone()) {
            return Err("example IDs must be nonempty and unique".into());
        }
        operations.validate_prompt("evaluation", &row.text)?;
        let input_sha256 = lfg_core::digest_text(&row.text);
        let result = encoder.encode(&PreflightRequest {
            session_id: "evaluation".into(),
            revision: count,
            text: row.text,
        });
        let prediction = Prediction {
            schema_version: "lfg.evaluation-prediction.v1",
            id: row.id,
            input_sha256,
            producer_sha256: producer_sha256.clone(),
            model: result.model,
            intents: result.intents.into_iter().map(|item| item.label).collect(),
            capabilities: result
                .capabilities
                .into_iter()
                .map(|item| item.label)
                .collect(),
            risks: result.risks.into_iter().map(|item| item.label).collect(),
        };
        serde_json::to_writer(&mut output, &prediction)?;
        writeln!(output)?;
        count = count.checked_add(1).ok_or("too many examples")?;
    }
    if count == 0 {
        return Err("no examples supplied".into());
    }
    Ok(())
}
