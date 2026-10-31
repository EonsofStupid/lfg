//! Experimental offline inference only. This path cannot promote or route work.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tract::prelude::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(feature = "embedded-onnx")]
mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_onnx.rs"));
}

enum ArtifactSource<'a> {
    Directory(&'a Path),
    #[cfg(feature = "embedded-onnx")]
    Embedded,
}

impl<'a> ArtifactSource<'a> {
    fn read(&self, name: &str, limit: u64) -> Result<Cow<'a, [u8]>> {
        match self {
            Self::Directory(root) => Ok(Cow::Owned(bounded_file(&root.join(name), limit)?)),
            #[cfg(feature = "embedded-onnx")]
            Self::Embedded => {
                let bytes = embedded::FILES
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, bytes)| *bytes)
                    .ok_or("embedded artifact file missing")?;
                if bytes.len() as u64 > limit {
                    return Err("embedded artifact exceeds configured limit".into());
                }
                Ok(Cow::Borrowed(bytes))
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RuntimeConfig {
    schema_version: String,
    revision: String,
    cpu_threads: usize,
    max_manifest_bytes: u64,
    max_metadata_bytes: u64,
    max_model_bytes: u64,
    max_sequence_tokens: usize,
    max_labels: usize,
    max_examples: usize,
    max_prompt_bytes: usize,
}

#[derive(Deserialize)]
struct Artifact {
    schema_version: String,
    promotion_eligible: bool,
    max_length: usize,
    padding_side: String,
    truncation_side: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    prediction_config_sha256: String,
    taxonomy_sha256: String,
    files_sha256: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct PredictionConfig {
    schema_version: String,
    default_threshold: f64,
    label_thresholds: BTreeMap<String, f64>,
    batch_size: usize,
    cpu_threads: usize,
    max_input_bytes: usize,
}

#[derive(Deserialize)]
struct Taxonomy {
    schema_version: String,
    intents: Vec<String>,
    capabilities: Vec<String>,
    risks: Vec<String>,
}

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
    model: Value,
    intents: Vec<String>,
    capabilities: Vec<String>,
    risks: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scores: Option<Scores>,
}

#[derive(Serialize)]
struct Scores {
    semantics: &'static str,
    values: BTreeMap<String, f64>,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sigmoid(value: f32) -> f64 {
    f64::from(1.0_f32 / (1.0_f32 + (-value).exp()))
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
        return Err("artifact file is not a regular file or exceeds its configured limit".into());
    }
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("artifact file exceeds its configured limit".into());
    }
    Ok(bytes)
}

fn checked_file<'a>(
    source: &ArtifactSource<'a>,
    name: &str,
    limit: u64,
    manifest: &Artifact,
) -> Result<Cow<'a, [u8]>> {
    let expected = manifest
        .files_sha256
        .get(name)
        .ok_or("required artifact digest missing")?;
    if !valid_digest(expected) {
        return Err("invalid artifact digest".into());
    }
    let bytes = source.read(name, limit)?;
    if hash(&bytes) != *expected {
        return Err("artifact digest mismatch".into());
    }
    Ok(bytes)
}

fn labels(values: &[String]) -> Result<BTreeSet<String>> {
    let result: BTreeSet<_> = values.iter().cloned().collect();
    if result.len() != values.len() || result.iter().any(|s| s.trim().is_empty()) {
        return Err("labels must be nonempty and unique".into());
    }
    Ok(result)
}

fn load_config(path: &Path) -> Result<(RuntimeConfig, String)> {
    let bytes = std::fs::read(path)?;
    parse_config(&bytes)
}

fn parse_config(bytes: &[u8]) -> Result<(RuntimeConfig, String)> {
    let config: RuntimeConfig = serde_json::from_slice(bytes)?;
    if config.schema_version != "lfg.onnx-runtime.v1"
        || config.revision.trim().is_empty()
        || config.cpu_threads != 1
        || config.max_manifest_bytes == 0
        || config.max_metadata_bytes == 0
        || config.max_model_bytes == 0
        || config.max_sequence_tokens == 0
        || config.max_labels == 0
        || config.max_examples == 0
        || config.max_prompt_bytes == 0
    {
        return Err("invalid or unsupported ONNX runtime configuration".into());
    }
    Ok((config, hash(bytes)))
}

pub fn run(
    root: &Path,
    expected_manifest: &str,
    config_path: &Path,
    include_scores: bool,
) -> Result<()> {
    let (config, config_hash) = load_config(config_path)?;
    run_source(
        ArtifactSource::Directory(root),
        expected_manifest,
        config,
        config_hash,
        include_scores,
    )
}

#[cfg(feature = "embedded-onnx")]
pub fn run_embedded(config_path: Option<&Path>, include_scores: bool) -> Result<()> {
    let (config, config_hash) = match config_path {
        Some(path) => load_config(path)?,
        None => parse_config(include_bytes!("../config/onnx-runtime.example.json"))?,
    };
    run_source(
        ArtifactSource::Embedded,
        embedded::MANIFEST_SHA256,
        config,
        config_hash,
        include_scores,
    )
}

fn run_source(
    source: ArtifactSource<'_>,
    expected_manifest: &str,
    config: RuntimeConfig,
    config_hash: String,
    include_scores: bool,
) -> Result<()> {
    if !valid_digest(expected_manifest) {
        return Err("expected manifest SHA-256 must be lowercase hexadecimal".into());
    }
    let manifest_bytes = source.read("manifest.json", config.max_manifest_bytes)?;
    if hash(&manifest_bytes) != expected_manifest {
        return Err("manifest digest mismatch".into());
    }
    let artifact: Artifact = serde_json::from_slice(&manifest_bytes)?;
    if artifact.schema_version != "lfg.encoder-onnx-artifact.v2"
        || artifact.promotion_eligible
        || artifact.max_length == 0
        || artifact.max_length > config.max_sequence_tokens
        || !["left", "right"].contains(&artifact.padding_side.as_str())
        || !["left", "right"].contains(&artifact.truncation_side.as_str())
        || artifact.inputs.is_empty()
        || artifact.outputs != ["logits"]
    {
        return Err("unsupported ONNX artifact contract".into());
    }
    let mut unique_inputs = HashSet::new();
    if artifact.inputs.iter().any(|name| {
        !["input_ids", "token_type_ids", "attention_mask"].contains(&name.as_str())
            || !unique_inputs.insert(name)
    }) || !artifact.inputs.iter().any(|name| name == "input_ids")
    {
        return Err("unsupported ONNX inputs".into());
    }
    let tokenizer_bytes = checked_file(
        &source,
        "tokenizer.json",
        config.max_metadata_bytes,
        &artifact,
    )?;
    let tokenizer_json: Value = serde_json::from_slice(&tokenizer_bytes)?;
    let truncation = &tokenizer_json["truncation"];
    let padding = &tokenizer_json["padding"];
    if truncation["max_length"].as_u64() != Some(artifact.max_length as u64)
        || truncation["direction"]
            .as_str()
            .map(str::to_lowercase)
            .as_deref()
            != Some(artifact.truncation_side.as_str())
        || truncation["strategy"] != "LongestFirst"
        || truncation["stride"] != 0
        || padding["strategy"] != "BatchLongest"
        || !padding["pad_to_multiple_of"].is_null()
        || padding["direction"]
            .as_str()
            .map(str::to_lowercase)
            .as_deref()
            != Some(artifact.padding_side.as_str())
    {
        return Err("tokenizer settings differ from the supported artifact contract".into());
    }
    let tokenizer =
        tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| e.to_string())?;
    let model_labels: Vec<String> = serde_json::from_slice(&checked_file(
        &source,
        "labels.json",
        config.max_metadata_bytes,
        &artifact,
    )?)?;
    let label_set = labels(&model_labels)?;
    if label_set.is_empty() || label_set.len() > config.max_labels {
        return Err("model label count is invalid".into());
    }
    let prediction_bytes = checked_file(
        &source,
        "prediction-config.json",
        config.max_metadata_bytes,
        &artifact,
    )?;
    if hash(&prediction_bytes) != artifact.prediction_config_sha256 {
        return Err("prediction configuration identity differs from manifest".into());
    }
    let prediction: PredictionConfig = serde_json::from_slice(&prediction_bytes)?;
    if prediction.schema_version != "lfg.encoder-prediction-config.v1"
        || prediction.batch_size == 0
        || prediction.cpu_threads == 0
        || prediction.max_input_bytes == 0
        || !prediction.default_threshold.is_finite()
        || !(0.0..=1.0).contains(&prediction.default_threshold)
        || prediction
            .label_thresholds
            .values()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return Err("invalid prediction configuration".into());
    }
    let taxonomy_bytes = checked_file(
        &source,
        "taxonomy.json",
        config.max_metadata_bytes,
        &artifact,
    )?;
    if hash(&taxonomy_bytes) != artifact.taxonomy_sha256 {
        return Err("taxonomy identity differs from manifest".into());
    }
    let taxonomy: Taxonomy = serde_json::from_slice(&taxonomy_bytes)?;
    let intent = labels(&taxonomy.intents)?;
    let capability = labels(&taxonomy.capabilities)?;
    let risk = labels(&taxonomy.risks)?;
    let all: BTreeSet<_> = intent
        .union(&capability)
        .cloned()
        .collect::<BTreeSet<_>>()
        .union(&risk)
        .cloned()
        .collect();
    if taxonomy.schema_version != "lfg.encoder-evaluation-taxonomy.v1"
        || intent.is_empty()
        || capability.is_empty()
        || risk.is_empty()
        || all.len() != intent.len() + capability.len() + risk.len()
        || !label_set.is_subset(&all)
        || prediction
            .label_thresholds
            .keys()
            .any(|label| !all.contains(label))
    {
        return Err("invalid or incompatible taxonomy".into());
    }
    let model_bytes = checked_file(&source, "model.onnx", config.max_model_bytes, &artifact)?;
    // Loading from a buffer provides no model directory; tract rejects external tensor references.
    let mut model = tract::onnx()?.load_buffer(&model_bytes)?;
    if model.input_count()? != artifact.inputs.len() || model.output_count()? != 1 {
        return Err("ONNX input or output count differs from its manifest".into());
    }
    for (index, name) in artifact.inputs.iter().enumerate() {
        if model.input_name(index)? != *name {
            return Err("ONNX input names differ from its manifest".into());
        }
        model.set_input_fact(index, "1,S,i64")?;
    }
    let runnable = model.into_model()?.into_runnable()?;
    let binary_hash = {
        let mut stream = File::open(std::env::current_exe()?)?;
        let mut hasher = Sha256::new();
        io::copy(&mut stream, &mut hasher)?;
        format!("{:x}", hasher.finalize())
    };
    let identity = json!({"artifact":expected_manifest,"config":config_hash,"taxonomy":hash(&taxonomy_bytes),"runtime":binary_hash});
    let model_identity = json!({"id":"lfg-onnx-shadow","version":"1","sha256":lfg_core::digest_value(&identity),
        "artifact":expected_manifest,"config":identity["config"],"taxonomy":identity["taxonomy"],"runtime":identity["runtime"]});
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
        if seen.len() >= config.max_examples {
            return Err("example count exceeds configured limit".into());
        }
        if row.id.trim().is_empty() || !seen.insert(row.id.clone()) {
            return Err("example IDs must be nonempty and unique".into());
        }
        operations.validate_prompt("onnx-shadow", &row.text)?;
        if row.text.len() > config.max_prompt_bytes || row.text.len() > prediction.max_input_bytes {
            return Err("prompt exceeds ONNX input limit".into());
        }
        let encoded = tokenizer
            .encode(row.text.as_str(), true)
            .map_err(|e| e.to_string())?;
        let length = encoded.len();
        if length == 0 || length > artifact.max_length {
            return Err("tokenizer output exceeds artifact limit".into());
        }
        let inputs = artifact
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
        let outputs = runnable.run(inputs)?;
        if outputs.len() != 1 || outputs[0].shape()? != [1, model_labels.len()] {
            return Err("invalid ONNX output shape".into());
        }
        let logits = outputs[0].as_slice::<f32>()?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("non-finite ONNX output".into());
        }
        let chosen: BTreeSet<_> = model_labels
            .iter()
            .zip(logits)
            .filter_map(|(label, value)| {
                let threshold = prediction
                    .label_thresholds
                    .get(label)
                    .copied()
                    .unwrap_or(prediction.default_threshold);
                (sigmoid(*value) >= threshold).then_some(label.clone())
            })
            .collect();
        let result = Prediction {
            schema_version: "lfg.evaluation-prediction.v1",
            id: row.id,
            input_sha256: lfg_core::digest_text(&row.text),
            producer_sha256: binary_hash.clone(),
            model: model_identity.clone(),
            intents: chosen.intersection(&intent).cloned().collect(),
            capabilities: chosen.intersection(&capability).cloned().collect(),
            risks: chosen.intersection(&risk).cloned().collect(),
            scores: include_scores.then(|| Scores {
                semantics: "sigmoid-uncalibrated-v1",
                values: model_labels
                    .iter()
                    .cloned()
                    .zip(logits.iter().copied().map(sigmoid))
                    .collect(),
            }),
        };
        serde_json::to_writer(&mut output, &result)?;
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

    #[test]
    fn requires_valid_config_and_pinned_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("runtime.json");
        std::fs::write(&config, include_str!("../config/onnx-runtime.example.json")).unwrap();
        let (parsed, _) = load_config(&config).unwrap();
        assert_eq!(parsed.cpu_threads, 1);
        assert!(!valid_digest("bad"));
        std::fs::write(dir.path().join("manifest.json"), "{}").unwrap();
        assert!(
            run(dir.path(), &"0".repeat(64), &config, false)
                .unwrap_err()
                .to_string()
                .contains("manifest digest mismatch")
        );
        let mut value: Value =
            serde_json::from_str(include_str!("../config/onnx-runtime.example.json")).unwrap();
        value["cpuThreads"] = json!(2);
        std::fs::write(&config, value.to_string()).unwrap();
        assert!(load_config(&config).is_err());
    }

    #[test]
    fn rejects_changed_or_oversized_artifact_files() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, "changed").unwrap();
        let artifact = Artifact {
            schema_version: String::new(),
            promotion_eligible: false,
            max_length: 1,
            padding_side: "right".into(),
            truncation_side: "right".into(),
            inputs: vec![],
            outputs: vec![],
            prediction_config_sha256: String::new(),
            taxonomy_sha256: String::new(),
            files_sha256: BTreeMap::from([("tokenizer.json".into(), hash(b"original"))]),
        };
        assert!(
            checked_file(
                &ArtifactSource::Directory(dir.path()),
                "tokenizer.json",
                100,
                &artifact
            )
            .is_err()
        );
        assert!(bounded_file(&file, 1).is_err());
    }
}
