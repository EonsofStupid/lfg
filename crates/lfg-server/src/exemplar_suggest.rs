//! Offline suggestions from a pinned, unpromoted per-node exemplar pack.
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    io::{self, BufRead, Read, Write},
    path::Path,
};

use lfg_core::kernel::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::onnx_embedding::{self, Encoder, Source};
use lfg_runtime::OperationsConfig;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Config {
    schema_version: String,
    revision: String,
    pooling: String,
    max_tokens: usize,
    max_text_bytes: usize,
    batch_size: usize,
    cpu_threads: usize,
    max_exemplars: usize,
    max_validation_items: usize,
    max_dimensions: usize,
    max_pack_bytes: u64,
    max_prediction_bytes: usize,
    minimum_similarity: f64,
    minimum_margin: f64,
}

impl Config {
    fn parse(raw: &[u8]) -> Result<Self> {
        let value: Self = serde_json::from_slice(raw)?;
        if value.schema_version != "lfg.node-exemplar-config.v1"
            || value.revision.trim().is_empty()
            || value.pooling != "attention-mask-mean-l2"
            || value.max_tokens == 0
            || value.max_text_bytes == 0
            || value.batch_size == 0
            || value.cpu_threads == 0
            || value.max_exemplars == 0
            || value.max_validation_items == 0
            || value.max_dimensions == 0
            || value.max_pack_bytes == 0
            || value.max_prediction_bytes == 0
            || !value.minimum_similarity.is_finite()
            || !(-1.0..=1.0).contains(&value.minimum_similarity)
            || !value.minimum_margin.is_finite()
            || !(0.0..=2.0).contains(&value.minimum_margin)
        {
            return Err("invalid exemplar configuration".into());
        }
        Ok(value)
    }

    fn encoder_digest(&self) -> String {
        lfg_core::digest_value(&json!({"pooling":self.pooling,"maxTokens":self.max_tokens,
            "maxTextBytes":self.max_text_bytes,"batchSize":self.batch_size,"cpuThreads":self.cpu_threads}))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pack {
    schema_version: String,
    promotion_eligible: bool,
    node: String,
    snapshot_sha256: String,
    admission: Value,
    model: Value,
    environment: Value,
    encoder_versions: Value,
    build_config_sha256: String,
    encoder_config_sha256: String,
    options: Vec<String>,
    dimension: usize,
    exemplars: Vec<Exemplar>,
    builder_sha256: String,
    input_verifier_sha256: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Exemplar {
    item_sha256: String,
    choice: String,
    embedding: Vec<f64>,
}

fn unit_vector(vector: &[f64], dimension: usize) -> bool {
    vector.len() == dimension
        && vector.iter().all(|v| v.is_finite())
        && (vector.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() <= 0.0001
}

impl Pack {
    fn parse(
        raw: &[u8],
        expected_sha: &str,
        config: &Config,
        source: &Source,
        engine: &Engine,
    ) -> Result<Self> {
        if !onnx_embedding::valid_digest(expected_sha) || onnx_embedding::hash(raw) != expected_sha
        {
            return Err("exemplar pack digest mismatch".into());
        }
        let pack: Self = serde_json::from_slice(raw)?;
        let options = engine
            .choice_options(&pack.node)
            .ok_or("pack node is not a snapshot choice")?;
        let admission = &pack.admission;
        let valid_value_digest =
            |value: &Value| value.as_str().is_some_and(onnx_embedding::valid_digest);
        let sources_valid = admission["sources"].as_array().is_some_and(|items| {
            !items.is_empty()
                && items.iter().all(|item| {
                    valid_value_digest(&item["queueSha256"])
                        && valid_value_digest(&item["ledgerSha256"])
                })
        });
        if pack.schema_version != "lfg.node-exemplar-pack.v1"
            || pack.promotion_eligible
            || pack.snapshot_sha256 != engine.snapshot_digest()
            || pack.options != options
            || options.len() < 2
            || pack.dimension == 0
            || pack.dimension > config.max_dimensions
            || pack.dimension != source.manifest.dimension
            || pack.encoder_config_sha256 != config.encoder_digest()
            || pack.encoder_config_sha256 != source.manifest.encoder_config_sha256
            || pack.model["manifest_sha256"].as_str()
                != Some(source.manifest.source_model_manifest_sha256.as_str())
            || admission["schemaVersion"] != "lfg.node-admission-report.v1"
            || admission["admissionEligible"] != true
            || admission["qualificationEstablished"] != false
            || admission["node"] != pack.node
            || admission["snapshotSha256"] != pack.snapshot_sha256
            || admission["pending"] != 0
            || admission["supportGatePassed"] != true
            || admission["projectGroupGatePassed"] != true
            || admission["conflicts"]
                .as_array()
                .is_none_or(|v| !v.is_empty())
            || admission["approved"].as_u64().is_none_or(|v| v == 0)
            || !sources_valid
            || !valid_value_digest(&pack.environment["lock_sha256"])
            || !valid_value_digest(&pack.environment["installed_versions_sha256"])
            || pack.environment["locked_wheel_payload"]["schema_version"]
                != "lfg.installed-wheel-payload.v1"
            || !valid_value_digest(&pack.environment["locked_wheel_payload"]["payload_sha256"])
            || pack.environment["locked_wheel_payload"]["files"]
                .as_u64()
                .is_none_or(|v| v == 0)
            || ["torch", "transformers"].iter().any(|key| {
                pack.encoder_versions[*key]
                    .as_str()
                    .is_none_or(|v| v.trim().is_empty())
            })
            || ["model_id", "revision"].iter().any(|key| {
                pack.model[*key]
                    .as_str()
                    .is_none_or(|v| v.trim().is_empty())
            })
            || pack.model["files"].as_u64().is_none_or(|v| v == 0)
            || [
                &pack.build_config_sha256,
                &pack.builder_sha256,
                &pack.input_verifier_sha256,
            ]
            .iter()
            .any(|v| !onnx_embedding::valid_digest(v))
            || pack.exemplars.is_empty()
            || pack.exemplars.len() > config.max_exemplars
        {
            return Err("exemplar pack is incompatible or not admitted".into());
        }
        let mut seen = BTreeSet::new();
        let mut classes = BTreeSet::new();
        let mut counts = BTreeMap::<&str, u64>::new();
        for entry in &pack.exemplars {
            if !onnx_embedding::valid_digest(&entry.item_sha256)
                || !seen.insert(&entry.item_sha256)
                || !options.contains(&entry.choice)
                || !unit_vector(&entry.embedding, pack.dimension)
            {
                return Err("invalid exemplar identity, choice or vector".into());
            }
            classes.insert(entry.choice.clone());
            *counts.entry(&entry.choice).or_default() += 1;
        }
        if classes != options.into_iter().collect() {
            return Err("every option requires an exemplar".into());
        }
        if counts
            .iter()
            .any(|(choice, count)| admission["counts"]["train"][*choice].as_u64() != Some(*count))
        {
            return Err("exemplar counts differ from admission report".into());
        }
        Ok(pack)
    }

    fn choose(&self, query: &[f32], config: &Config) -> Result<Option<(String, f64, f64)>> {
        let query: Vec<f64> = query.iter().map(|v| f64::from(*v)).collect();
        if !unit_vector(&query, self.dimension) {
            return Err("query embedding is not a finite unit vector".into());
        }
        let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
        for entry in &self.exemplars {
            let similarity = query
                .iter()
                .zip(&entry.embedding)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1.0, 1.0);
            scores
                .entry(&entry.choice)
                .and_modify(|v| *v = v.max(similarity))
                .or_insert(similarity);
        }
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let margin = ranked[0].1 - ranked[1].1;
        if ranked[0].1 < config.minimum_similarity || margin <= config.minimum_margin {
            return Ok(None);
        }
        Ok(Some((ranked[0].0.to_owned(), ranked[0].1, margin)))
    }
}

#[derive(Deserialize)]
struct Input {
    id: String,
    text: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Prediction<'a> {
    schema_version: &'static str,
    item_sha256: &'a str,
    input_sha256: String,
    snapshot_sha256: &'a str,
    choice: Option<&'a str>,
}

pub struct Options<'a> {
    pub artifact: &'a Path,
    pub manifest_sha256: &'a str,
    pub runtime_config: &'a Path,
    pub exemplar_pack: &'a Path,
    pub pack_sha256: &'a str,
    pub exemplar_config: &'a Path,
}

/// An unpromoted suggestion and the deterministic result it extends.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeResponse {
    pub schema_version: &'static str,
    pub input_sha256: String,
    pub snapshot_sha256: String,
    pub encoder_manifest_sha256: String,
    pub pack_sha256: String,
    pub exemplar_config_sha256: String,
    pub runtime_config_sha256: String,
    pub promotion_eligible: bool,
    pub baseline: Value,
    pub extension_status: &'static str,
    pub suggestion: Option<Suggestion>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub choice: String,
    pub similarity: f64,
    pub margin: f64,
    pub source: &'static str,
}

/// Snapshot-bound experimental extension. The encoder loads only on escalation.
pub struct NodeSuggester<'a> {
    engine: &'a Engine,
    config: Config,
    source: Source,
    pack: Pack,
    operations: OperationsConfig,
    config_sha: String,
    pack_sha: String,
    encoder: Option<Encoder>,
}

impl<'a> NodeSuggester<'a> {
    /// Open the small manifests and pack; model weights remain unloaded.
    pub fn open(
        args: &Options<'_>,
        engine: &'a Engine,
        operations: OperationsConfig,
    ) -> Result<Self> {
        let config_raw = std::fs::read(args.exemplar_config)?;
        let config = Config::parse(&config_raw)?;
        let source = Source::open(args.artifact, args.manifest_sha256, args.runtime_config)?;
        let raw_pack = onnx_embedding::bounded_file(args.exemplar_pack, config.max_pack_bytes)?;
        let pack = Pack::parse(&raw_pack, args.pack_sha256, &config, &source, engine)?;
        Ok(Self {
            engine,
            config,
            source,
            pack,
            operations,
            config_sha: onnx_embedding::hash(&config_raw),
            pack_sha: args.pack_sha256.to_owned(),
            encoder: None,
        })
    }

    /// Use explicitly built-in artifacts, with optional runtime configuration overrides.
    #[cfg(feature = "embedded-node")]
    pub fn open_embedded(
        engine: &'a Engine,
        operations: OperationsConfig,
        runtime_override: Option<&Path>,
        config_override: Option<&Path>,
    ) -> Result<Self> {
        let runtime = match runtime_override {
            Some(path) => std::borrow::Cow::Owned(std::fs::read(path)?),
            None => std::borrow::Cow::Borrowed(embedded_file("runtime.json")?),
        };
        let config_raw = match config_override {
            Some(path) => std::borrow::Cow::Owned(std::fs::read(path)?),
            None => std::borrow::Cow::Borrowed(embedded_file("exemplar-config.json")?),
        };
        let config = Config::parse(&config_raw)?;
        let source = Source::open_embedded(&runtime)?;
        let raw_pack = embedded_file("pack.json")?;
        if raw_pack.len() as u64 > config.max_pack_bytes {
            return Err("embedded exemplar pack exceeds configured limit".into());
        }
        let pack = Pack::parse(
            raw_pack,
            crate::embedded_node::PACK_SHA256,
            &config,
            &source,
            engine,
        )?;
        Ok(Self {
            engine,
            config,
            source,
            pack,
            operations,
            config_sha: onnx_embedding::hash(&config_raw),
            pack_sha: crate::embedded_node::PACK_SHA256.to_owned(),
            encoder: None,
        })
    }

    /// Evaluate one request; suggestions are advisory and never replace the baseline.
    pub fn suggest(&mut self, text: &str) -> Result<NodeResponse> {
        self.operations
            .validate_prompt("node-suggestion-probe", text)?;
        let baseline = serde_json::to_value(
            self.engine
                .evaluate(&BTreeMap::from([("prompt".into(), text.to_owned())]))?,
        )?;
        let eligible = baseline["state"] == "escalate" && baseline["atNode"] == self.pack.node;
        let suggestion = if eligible {
            if text.len() > self.config.max_text_bytes
                || text.len() > self.source.config.max_prompt_bytes
            {
                return Err("input exceeds exemplar text limit".into());
            }
            if self.encoder.is_none() {
                self.encoder = Some(self.source.load()?);
            }
            let vector = self
                .encoder
                .as_ref()
                .expect("encoder was initialized")
                .embed(text)?;
            self.pack
                .choose(&vector, &self.config)?
                .map(|(choice, similarity, margin)| Suggestion {
                    choice,
                    similarity,
                    margin,
                    source: "exemplar",
                })
        } else {
            None
        };
        Ok(NodeResponse {
            schema_version: "lfg.node-suggestion.v1",
            input_sha256: lfg_core::digest_text(text),
            snapshot_sha256: self.engine.snapshot_digest().to_owned(),
            encoder_manifest_sha256: self.source.manifest_sha.clone(),
            pack_sha256: self.pack_sha.clone(),
            exemplar_config_sha256: self.config_sha.clone(),
            runtime_config_sha256: self.source.config_sha.clone(),
            promotion_eligible: false,
            baseline,
            extension_status: if !eligible {
                "not_applicable"
            } else if suggestion.is_some() {
                "suggested"
            } else {
                "abstained"
            },
            suggestion,
        })
    }
}

#[cfg(feature = "embedded-node")]
fn embedded_file(name: &str) -> Result<&'static [u8]> {
    crate::embedded_node::FILES
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, bytes)| *bytes)
        .ok_or_else(|| "embedded node file missing".into())
}

/// Selected snapshot bytes for a binary built with `embedded-node`.
#[cfg(feature = "embedded-node")]
pub fn embedded_snapshot() -> Result<&'static [u8]> {
    embedded_file("snapshot.json")
}

pub fn run(args: Options<'_>, engine: &Engine, prediction_only: bool) -> Result<()> {
    let suggester = NodeSuggester::open(&args, engine, crate::configured_operations()?)?;
    stream(suggester, prediction_only)
}

#[cfg(feature = "embedded-node")]
pub fn run_embedded(
    runtime: Option<&Path>,
    config: Option<&Path>,
    prediction_only: bool,
) -> Result<()> {
    let engine = Engine::from_json(embedded_snapshot()?)?;
    let suggester =
        NodeSuggester::open_embedded(&engine, crate::configured_operations()?, runtime, config)?;
    stream(suggester, prediction_only)
}

fn stream(mut suggester: NodeSuggester<'_>, prediction_only: bool) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut line = String::new();
    let mut seen = HashSet::new();
    let mut output_bytes = 0usize;
    loop {
        line.clear();
        let read = reader
            .by_ref()
            .take((suggester.operations.max_http_body_bytes() as u64).saturating_add(1))
            .read_line(&mut line)?;
        if read == 0 {
            break;
        }
        if read > suggester.operations.max_http_body_bytes() {
            return Err("input line exceeds configured limit".into());
        }
        if line.trim().is_empty() {
            continue;
        }
        let input: Input = serde_json::from_str(&line)?;
        if input.id.trim().is_empty()
            || !seen.insert(input.id.clone())
            || seen.len() > suggester.source.config.max_examples
            || seen.len() > suggester.config.max_validation_items
        {
            return Err("empty or duplicate ID, or example limit exceeded".into());
        }
        let response = suggester.suggest(&input.text)?;
        let choice = response
            .suggestion
            .as_ref()
            .map(|item| item.choice.as_str());
        let row = if prediction_only {
            serde_json::to_value(Prediction {
                schema_version: "lfg.node-prediction.v1",
                item_sha256: &input.id,
                input_sha256: lfg_core::digest_text(&input.text),
                snapshot_sha256: suggester.engine.snapshot_digest(),
                choice,
            })?
        } else {
            json!({"schemaVersion":"lfg.node-suggestion-probe.v1","id":input.id,
                "inputSha256":lfg_core::digest_text(&input.text),"snapshotSha256":suggester.engine.snapshot_digest(),
                "encoderManifestSha256":suggester.source.manifest_sha,"packSha256":suggester.pack_sha,
                "exemplarConfigSha256":suggester.config_sha,"runtimeConfigSha256":suggester.source.config_sha,
                "baseline":response.baseline,"extensionStatus":response.extension_status,
                "suggestion":response.suggestion,
                "promotionEligible":false})
        };
        let mut encoded = serde_json::to_vec(&row)?;
        encoded.push(b'\n');
        output_bytes = output_bytes
            .checked_add(encoded.len())
            .ok_or("output byte count overflow")?;
        if output_bytes > suggester.config.max_prediction_bytes {
            return Err("prediction output exceeds configured limit".into());
        }
        output.write_all(&encoded)?;
        output.flush()?;
    }
    if seen.is_empty() {
        return Err("no examples supplied".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, Config, Source, Engine, Value) {
        let dir = tempfile::tempdir().unwrap();
        let config = Config::parse(include_bytes!(
            "../../../ml/lfg-encoder/node-exemplar-config.example.json"
        ))
        .unwrap();
        let runtime = dir.path().join("runtime.json");
        std::fs::write(
            &runtime,
            include_bytes!("../config/node-onnx-runtime.example.json"),
        )
        .unwrap();
        let manifest = json!({"schemaVersion":"lfg.node-onnx-encoder.v1", "promotionEligible":false,
            "modelSha256":"a".repeat(64), "tokenizerSha256":"b".repeat(64),
            "sourceModelManifestSha256":"c".repeat(64), "encoderConfigSha256":config.encoder_digest(),
            "exportConfigSha256":"d".repeat(64), "inputs":["input_ids","attention_mask"],
            "output":"embedding", "dimension":2, "maxTokens":256, "pooling":"attention-mask-mean-l2"});
        let raw = serde_json::to_vec(&manifest).unwrap();
        std::fs::write(dir.path().join("manifest.json"), &raw).unwrap();
        let source = Source::open(dir.path(), &onnx_embedding::hash(&raw), &runtime).unwrap();
        let engine = Engine::bundled().unwrap();
        let pack = json!({"schemaVersion":"lfg.node-exemplar-pack.v1", "promotionEligible":false,
            "node":"operation", "snapshotSha256":engine.snapshot_digest(),
            "admission":{"schemaVersion":"lfg.node-admission-report.v1", "admissionEligible":true,
                "qualificationEstablished":false, "node":"operation", "snapshotSha256":engine.snapshot_digest(),
                "pending":0, "approved":6, "supportGatePassed":true, "projectGroupGatePassed":true,
                "conflicts":[], "counts":{"train":{"chat":1,"tool_call":1}},
                "sources":[{"queueSha256":"1".repeat(64),"ledgerSha256":"2".repeat(64)}]},
            "model":{"manifest_sha256":"c".repeat(64),"model_id":"fixture","revision":"fixture","files":1},
            "environment":{"lock_sha256":"3".repeat(64),"installed_versions_sha256":"4".repeat(64),
                "locked_wheel_payload":{"schema_version":"lfg.installed-wheel-payload.v1","payload_sha256":"5".repeat(64),"files":1}},
            "encoderVersions":{"torch":"fixture","transformers":"fixture"},
            "buildConfigSha256":"d".repeat(64), "encoderConfigSha256":config.encoder_digest(),
            "options":["chat","tool_call"], "dimension":2,
            "exemplars":[{"itemSha256":"a".repeat(64),"choice":"chat","embedding":[1.0,0.0]},
                         {"itemSha256":"b".repeat(64),"choice":"tool_call","embedding":[0.0,1.0]}],
            "builderSha256":"e".repeat(64), "inputVerifierSha256":"f".repeat(64)});
        (dir, config, source, engine, pack)
    }

    #[test]
    fn similarity_is_unpromoted_and_dynamic_thresholds_preserve_encoder_identity() {
        let (_dir, mut config, source, engine, value) = fixture();
        assert_eq!(
            config.encoder_digest(),
            "7e1653c34cef49abcb46c18571d8fa18ea597521934a6d19279fc32ce0fb2359"
        );
        let raw = serde_json::to_vec(&value).unwrap();
        let pack =
            Pack::parse(&raw, &onnx_embedding::hash(&raw), &config, &source, &engine).unwrap();
        assert_eq!(
            pack.choose(&[1.0, 0.0], &config).unwrap().unwrap().0,
            "chat"
        );
        assert!(
            pack.choose(&[std::f32::consts::FRAC_1_SQRT_2; 2], &config)
                .unwrap()
                .is_none()
        );
        let old_digest = config.encoder_digest();
        config.minimum_margin = 2.0;
        assert_eq!(old_digest, config.encoder_digest());
        assert!(pack.choose(&[1.0, 0.0], &config).unwrap().is_none());
        assert!(pack.choose(&[2.0, 0.0], &config).is_err());
    }

    #[test]
    fn rejects_changed_packs_and_incompatible_admission() {
        let (_dir, config, source, engine, mut value) = fixture();
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(Pack::parse(&raw, &"0".repeat(64), &config, &source, &engine).is_err());
        value["admission"]["pending"] = json!(1);
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(Pack::parse(&raw, &onnx_embedding::hash(&raw), &config, &source, &engine).is_err());
        value["admission"]["pending"] = json!(0);
        value["admission"]["counts"]["train"]["chat"] = json!(2);
        let raw = serde_json::to_vec(&value).unwrap();
        assert!(Pack::parse(&raw, &onnx_embedding::hash(&raw), &config, &source, &engine).is_err());
    }
}
