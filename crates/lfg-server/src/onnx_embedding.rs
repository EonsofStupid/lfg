//! Offline, unpromoted native encoder probe; it never selects or routes work.
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tract::prelude::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeConfig {
    schema_version: String,
    revision: String,
    cpu_threads: usize,
    max_manifest_bytes: u64,
    max_tokenizer_bytes: u64,
    max_model_bytes: u64,
    max_sequence_tokens: usize,
    max_dimensions: usize,
    max_examples: usize,
    max_prompt_bytes: usize,
    unit_norm_tolerance: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    promotion_eligible: bool,
    model_sha256: String,
    tokenizer_sha256: String,
    source_model_manifest_sha256: String,
    encoder_config_sha256: String,
    export_config_sha256: String,
    inputs: Vec<String>,
    output: String,
    dimension: usize,
    max_tokens: usize,
    pooling: String,
}

#[derive(Deserialize)]
struct Example {
    id: String,
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Embedding<'a> {
    schema_version: &'static str,
    id: &'a str,
    input_sha256: String,
    manifest_sha256: &'a str,
    runtime_config_sha256: &'a str,
    dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() > limit {
        return Err("artifact file is not regular or exceeds its configured limit".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("artifact file exceeds configured limit".into());
    }
    Ok(bytes)
}

pub fn run(
    root: &Path,
    expected_manifest: &str,
    config_path: &Path,
    include_embedding: bool,
) -> Result<()> {
    let config_raw = std::fs::read(config_path)?;
    let config: RuntimeConfig = serde_json::from_slice(&config_raw)?;
    if config.schema_version != "lfg.node-onnx-runtime.v1"
        || config.revision.trim().is_empty()
        || config.cpu_threads != 1
        || config.max_manifest_bytes == 0
        || config.max_tokenizer_bytes == 0
        || config.max_model_bytes == 0
        || config.max_sequence_tokens == 0
        || config.max_dimensions == 0
        || config.max_examples == 0
        || config.max_prompt_bytes == 0
        || !config.unit_norm_tolerance.is_finite()
        || !(0.0..1.0).contains(&config.unit_norm_tolerance)
        || !valid_digest(expected_manifest)
    {
        return Err("invalid node encoder runtime configuration or manifest digest".into());
    }
    let config_sha = hash(&config_raw);
    let manifest_raw = bounded_file(&root.join("manifest.json"), config.max_manifest_bytes)?;
    if hash(&manifest_raw) != expected_manifest {
        return Err("node encoder manifest digest mismatch".into());
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_raw)?;
    if manifest.schema_version != "lfg.node-onnx-encoder.v1"
        || manifest.promotion_eligible
        || manifest.output != "embedding"
        || manifest.pooling != "attention-mask-mean-l2"
        || manifest.dimension == 0
        || manifest.dimension > config.max_dimensions
        || manifest.max_tokens == 0
        || manifest.max_tokens > config.max_sequence_tokens
        || [
            &manifest.model_sha256,
            &manifest.tokenizer_sha256,
            &manifest.source_model_manifest_sha256,
            &manifest.encoder_config_sha256,
            &manifest.export_config_sha256,
        ]
        .iter()
        .any(|v| !valid_digest(v))
    {
        return Err("unsupported node encoder manifest".into());
    }
    let mut unique = HashSet::new();
    if manifest.inputs.is_empty()
        || manifest.inputs.iter().any(|name| {
            !["input_ids", "token_type_ids", "attention_mask"].contains(&name.as_str())
                || !unique.insert(name)
        })
        || !manifest.inputs.iter().any(|name| name == "input_ids")
        || !manifest.inputs.iter().any(|name| name == "attention_mask")
    {
        return Err("unsupported node encoder inputs".into());
    }
    let tokenizer_bytes = bounded_file(&root.join("tokenizer.json"), config.max_tokenizer_bytes)?;
    if hash(&tokenizer_bytes) != manifest.tokenizer_sha256 {
        return Err("tokenizer digest mismatch".into());
    }
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| e.to_string())?;
    if tokenizer.get_truncation().is_some() || tokenizer.get_padding().is_some() {
        return Err("node tokenizer must not truncate or pad automatically".into());
    }
    let model_bytes = bounded_file(&root.join("model.onnx"), config.max_model_bytes)?;
    if hash(&model_bytes) != manifest.model_sha256 {
        return Err("node encoder model digest mismatch".into());
    }
    // Loading from a buffer does not provide a directory for external tensor files.
    let mut model = tract::onnx()?.load_buffer(&model_bytes)?;
    if model.input_count()? != manifest.inputs.len() || model.output_count()? != 1 {
        return Err("ONNX input or output count differs from manifest".into());
    }
    for (index, name) in manifest.inputs.iter().enumerate() {
        if model.input_name(index)? != *name {
            return Err("ONNX input name differs from manifest".into());
        }
        model.set_input_fact(index, "1,S,i64")?;
    }
    let runnable = model.into_model()?.into_runnable()?;
    let operations = lfg_server::configured_operations()?;
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut line = String::new();
    let mut seen = HashSet::new();
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take((operations.max_http_body_bytes() as u64).saturating_add(1))
            .read_line(&mut line)?;
        if read == 0 {
            break;
        }
        if read > operations.max_http_body_bytes() {
            return Err("input line exceeds configured limit".into());
        }
        if line.trim().is_empty() {
            continue;
        }
        let row: Example = serde_json::from_str(&line)?;
        if seen.len() >= config.max_examples
            || row.id.trim().is_empty()
            || !seen.insert(row.id.clone())
        {
            return Err("example limit exceeded or duplicate or empty ID".into());
        }
        operations.validate_prompt("node-encoder-probe", &row.text)?;
        if row.text.len() > config.max_prompt_bytes {
            return Err("prompt exceeds configured byte limit".into());
        }
        let encoded = tokenizer
            .encode(row.text.as_str(), true)
            .map_err(|e| e.to_string())?;
        let length = encoded.len();
        if length == 0 || length > manifest.max_tokens {
            return Err("input exceeds encoder token limit".into());
        }
        let inputs = manifest
            .inputs
            .iter()
            .map(|name| {
                let values = match name.as_str() {
                    "input_ids" => encoded.get_ids(),
                    "token_type_ids" => encoded.get_type_ids(),
                    "attention_mask" => encoded.get_attention_mask(),
                    _ => unreachable!(),
                };
                Tensor::from_slice(
                    &[1, length],
                    &values.iter().map(|&v| i64::from(v)).collect::<Vec<_>>(),
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let result = runnable.run(inputs)?;
        if result.len() != 1 || result[0].shape()? != [1, manifest.dimension] {
            return Err("invalid node encoder output shape".into());
        }
        let vector = result[0].as_slice::<f32>()?;
        if vector.iter().any(|value| !value.is_finite()) {
            return Err("non-finite node embedding".into());
        }
        let norm = vector
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > config.unit_norm_tolerance {
            return Err("node embedding is not normalized".into());
        }
        serde_json::to_writer(
            &mut output,
            &Embedding {
                schema_version: "lfg.node-embedding-probe.v1",
                id: &row.id,
                input_sha256: lfg_core::digest_text(&row.text),
                manifest_sha256: expected_manifest,
                runtime_config_sha256: &config_sha,
                dimension: manifest.dimension,
                embedding: include_embedding.then(|| vector.to_vec()),
            },
        )?;
        writeln!(output)?;
    }
    if seen.is_empty() {
        return Err("no examples supplied".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_wrong_manifest_and_duplicate_inputs_before_loading_model() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("runtime.json");
        std::fs::write(
            &config,
            include_str!("../config/node-onnx-runtime.example.json"),
        )
        .unwrap();
        let manifest = json!({"schemaVersion":"lfg.node-onnx-encoder.v1","promotionEligible":false,
            "modelSha256":"a".repeat(64),"tokenizerSha256":"b".repeat(64),
            "sourceModelManifestSha256":"c".repeat(64),"encoderConfigSha256":"d".repeat(64),
            "exportConfigSha256":"e".repeat(64),"inputs":["input_ids","input_ids","attention_mask"],
            "output":"embedding","dimension":384,"maxTokens":256,"pooling":"attention-mask-mean-l2"});
        let raw = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(dir.path().join("manifest.json"), &raw).unwrap();
        assert!(
            run(dir.path(), &"0".repeat(64), &config, false)
                .unwrap_err()
                .to_string()
                .contains("digest mismatch")
        );
        assert!(
            run(dir.path(), &hash(&raw), &config, false)
                .unwrap_err()
                .to_string()
                .contains("unsupported node encoder inputs")
        );
    }

    #[test]
    fn bounded_files_reject_oversize() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, b"abcdef").unwrap();
        assert!(bounded_file(&path, 5).is_err());
        assert_eq!(bounded_file(&path, 6).unwrap(), b"abcdef");
        assert!(valid_digest(&"a".repeat(64)));
        assert!(!valid_digest(&"A".repeat(64)));
    }
}
