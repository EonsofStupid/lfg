//! The trained model that chooses an assessment's bracket. It is loaded once,
//! checked against the manifest digest the assessment configuration pins, and
//! run in process. It only scores text; it never routes or dispatches work.

use lfg_core::assessment::{ModelOutput, ModelScoreBinding};

/// Scores one prompt. Implementations must be safe to share across threads.
pub(crate) trait LevelScorer: Send + Sync {
    fn score(&self, text: &str) -> Result<ModelOutput, String>;
    /// How many prompts may be scored at once; more are refused as busy.
    fn max_concurrent(&self) -> usize;
}

/// Loads the scorer named by `LFG_MODEL_SCORE_CONFIG` for a configuration that binds one.
pub(crate) fn load(binding: &ModelScoreBinding) -> Result<std::sync::Arc<dyn LevelScorer>, String> {
    #[cfg(feature = "onnx-shadow")]
    {
        let path = std::env::var_os("LFG_MODEL_SCORE_CONFIG").ok_or(
            "the assessment configuration binds a trained model; set LFG_MODEL_SCORE_CONFIG to its runtime configuration",
        )?;
        let scorer = native::TractLevelScorer::open(std::path::Path::new(&path), binding)
            .map_err(|error| error.to_string())?;
        Ok(std::sync::Arc::new(scorer))
    }
    #[cfg(not(feature = "onnx-shadow"))]
    {
        let _ = binding;
        Err("the assessment configuration binds a trained model, but this build has no model runtime; build lfg-server with the onnx-shadow feature".into())
    }
}

#[cfg(feature = "onnx-shadow")]
pub(crate) mod native {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use serde::Deserialize;
    use tract::prelude::*;

    use super::LevelScorer;
    use crate::onnx_embedding::{bounded_file, hash, valid_digest};
    use lfg_core::assessment::{ModelOutput, ModelScoreBinding};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct RuntimeConfig {
        schema_version: String,
        revision: String,
        directory: PathBuf,
        max_manifest_bytes: u64,
        max_input_spec_bytes: u64,
        max_tokenizer_bytes: u64,
        max_model_bytes: u64,
        max_prompt_bytes: usize,
        max_concurrent: usize,
        /// Threads for one prompt's matrix work. Size it from measurements and
        /// what else shares the machine.
        cpu_threads: usize,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Manifest {
        schema_version: String,
        id: String,
        promotion_eligible: bool,
        model_file: String,
        model_sha256: String,
        tokenizer_sha256: String,
        input_spec_sha256: String,
        inputs: Vec<String>,
        output: String,
        levels: usize,
        /// Where the weights came from and how they were trained; recorded, not interpreted.
        #[serde(default)]
        provenance: serde_json::Value,
    }

    /// How a prompt becomes model input: a fixed prefix holding the question and
    /// its option markers, then the prompt's tokens, then the separator.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InputSpec {
        schema: String,
        prefix_ids: Vec<i64>,
        marker_positions: Vec<usize>,
        max_tokens: usize,
        room: usize,
        sep_id: i64,
        pad_id: i64,
        mask_token: String,
        output: String,
        levels: Vec<String>,
        tokenizer_sha256: String,
    }

    pub(crate) struct TractLevelScorer {
        tokenizer: tokenizers::Tokenizer,
        runnable: Runnable,
        prefix: Vec<i64>,
        room: usize,
        sep: i64,
        mask_token: String,
        levels: usize,
        max_prompt_bytes: usize,
        max_concurrent: usize,
        executor: tract_linalg::multithread::Executor,
    }

    fn safe_file_name(name: &str) -> bool {
        !name.is_empty()
            && name.len() <= 128
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            && !name.starts_with('.')
    }

    impl TractLevelScorer {
        pub(crate) fn open(config_path: &Path, binding: &ModelScoreBinding) -> Result<Self> {
            let config: RuntimeConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
            if config.schema_version != "lfg.model-score-runtime.v1"
                || config.revision.trim().is_empty()
                || [
                    config.max_manifest_bytes,
                    config.max_input_spec_bytes,
                    config.max_tokenizer_bytes,
                    config.max_model_bytes,
                ]
                .contains(&0)
                || config.max_prompt_bytes == 0
                || config.max_concurrent == 0
                || !(1..=256).contains(&config.cpu_threads)
            {
                return Err("invalid model score runtime configuration".into());
            }
            let root = &config.directory;
            let manifest_raw = bounded_file(&root.join("manifest.json"), config.max_manifest_bytes)
                .map_err(|e| e.to_string())?;
            if hash(&manifest_raw) != binding.manifest_sha256 {
                return Err(
                    "model manifest digest differs from the assessment configuration".into(),
                );
            }
            let manifest: Manifest = serde_json::from_slice(&manifest_raw)?;
            if manifest.schema_version != "lfg.model-score-manifest.v1"
                || manifest.id != binding.id
                || manifest.promotion_eligible
                || !safe_file_name(&manifest.model_file)
                || manifest.inputs != ["input_ids", "attention_mask"]
                || manifest.output != "level_probabilities"
                || manifest.levels != binding.level_brackets.len()
                || ![
                    &manifest.model_sha256,
                    &manifest.tokenizer_sha256,
                    &manifest.input_spec_sha256,
                ]
                .iter()
                .all(|digest| valid_digest(digest))
            {
                return Err("unsupported model score manifest".into());
            }
            let _ = &manifest.provenance;
            let spec_raw = bounded_file(&root.join("input.json"), config.max_input_spec_bytes)
                .map_err(|e| e.to_string())?;
            if hash(&spec_raw) != manifest.input_spec_sha256 {
                return Err("model input specification digest mismatch".into());
            }
            let spec: InputSpec = serde_json::from_slice(&spec_raw)?;
            if spec.schema != "lfg.dev-model-input.v1"
                || spec.prefix_ids.is_empty()
                || spec.marker_positions.len() != manifest.levels
                || spec
                    .marker_positions
                    .iter()
                    .any(|&p| p >= spec.prefix_ids.len())
                || spec.levels.len() != manifest.levels
                || spec.prefix_ids.len() + spec.room + 1 != spec.max_tokens
                || spec.mask_token.is_empty()
                || spec.tokenizer_sha256 != manifest.tokenizer_sha256
                || !spec.output.starts_with("level_probabilities")
            {
                return Err("unsupported model input specification".into());
            }
            let _ = spec.pad_id;
            let tokenizer_bytes =
                bounded_file(&root.join("tokenizer.json"), config.max_tokenizer_bytes)
                    .map_err(|e| e.to_string())?;
            if hash(&tokenizer_bytes) != manifest.tokenizer_sha256 {
                return Err("tokenizer digest mismatch".into());
            }
            let mut tokenizer =
                tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| e.to_string())?;
            drop(tokenizer_bytes);
            // The prefix and the length limit are applied here, never by the tokenizer.
            tokenizer
                .with_truncation(None)
                .map_err(|e| e.to_string())?
                .with_padding(None);
            let model_bytes =
                bounded_file(&root.join(&manifest.model_file), config.max_model_bytes)
                    .map_err(|e| e.to_string())?;
            if hash(&model_bytes) != manifest.model_sha256 {
                return Err("model digest mismatch".into());
            }
            let mut model = tract::onnx()?.load_buffer(&model_bytes)?;
            drop(model_bytes);
            if model.input_count()? != 2 || model.output_count()? != 1 {
                return Err("ONNX input or output count differs from manifest".into());
            }
            for (index, name) in manifest.inputs.iter().enumerate() {
                if model.input_name(index)? != *name {
                    return Err("ONNX input name differs from manifest".into());
                }
                model.set_input_fact(index, "1,S,i64")?;
            }
            let runnable = model.into_model()?.into_runnable()?;
            Ok(Self {
                tokenizer,
                runnable,
                prefix: spec.prefix_ids,
                room: spec.room,
                sep: spec.sep_id,
                mask_token: spec.mask_token,
                levels: manifest.levels,
                max_prompt_bytes: config.max_prompt_bytes,
                max_concurrent: config.max_concurrent,
                executor: if config.cpu_threads > 1 {
                    tract_linalg::multithread::Executor::multithread_with_name(
                        config.cpu_threads,
                        "lfg-level-score",
                    )
                } else {
                    tract_linalg::multithread::Executor::SingleThread
                },
            })
        }

        fn run(&self, text: &str) -> Result<ModelOutput> {
            if text.trim().is_empty() || text.len() > self.max_prompt_bytes {
                return Err("prompt is empty or exceeds the model's byte limit".into());
            }
            let started = Instant::now();
            let cleaned = text.replace(&self.mask_token, " ");
            let encoded = self
                .tokenizer
                .encode(cleaned.as_str(), false)
                .map_err(|e| e.to_string())?;
            let tokens = encoded.get_ids();
            let truncated = tokens.len() > self.room;
            let mut ids = self.prefix.clone();
            ids.extend(tokens.iter().take(self.room).map(|&id| i64::from(id)));
            ids.push(self.sep);
            let length = ids.len();
            let input_ids = Tensor::from_slice(&[1, length], &ids)?;
            let mask = Tensor::from_slice(&[1, length], &vec![1i64; length])?;
            let result =
                tract_linalg::multithread::multithread_tract_scope(self.executor.clone(), || {
                    self.runnable.run(vec![input_ids, mask])
                })?;
            if result.len() != 1 || result[0].shape()? != [1, self.levels] {
                return Err("invalid model output shape".into());
            }
            let probabilities: Vec<f64> = result[0]
                .as_slice::<f32>()?
                .iter()
                .map(|p| f64::from(*p))
                .collect();
            Ok(ModelOutput {
                probabilities,
                elapsed_micros: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                truncated,
            })
        }
    }

    impl LevelScorer for TractLevelScorer {
        fn score(&self, text: &str) -> std::result::Result<ModelOutput, String> {
            self.run(text).map_err(|error| error.to_string())
        }

        fn max_concurrent(&self) -> usize {
            self.max_concurrent
        }
    }
}

#[cfg(all(test, feature = "onnx-shadow"))]
mod tests {
    use super::native::TractLevelScorer;
    use super::*;

    const FIXTURE_MANIFEST: &str =
        "de3df884873479acd677322b4bc1abfeb67bee60802a4c06ffa18905d80fb849";
    const LEVEL_WEIGHTS: [f64; 5] = [-0.1, -0.05, 0.0, 0.05, 0.1];

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/level-score")
    }

    fn binding(manifest: &str) -> ModelScoreBinding {
        ModelScoreBinding {
            id: "level-score-fixture".into(),
            manifest_sha256: manifest.into(),
            level_brackets: ["trivial", "simple", "moderate", "complex", "major"]
                .map(String::from)
                .to_vec(),
            level_cut_points: vec![0.5, 1.5, 2.5, 3.5],
        }
    }

    fn runtime(
        dir: &std::path::Path,
        root: &std::path::Path,
        threads: usize,
    ) -> std::path::PathBuf {
        let path = dir.join("runtime.json");
        std::fs::write(
            &path,
            serde_json::json!({"schemaVersion":"lfg.model-score-runtime.v1","revision":"test",
                "directory": root, "maxManifestBytes": 65536, "maxInputSpecBytes": 65536,
                "maxTokenizerBytes": 65536, "maxModelBytes": 65536, "maxPromptBytes": 4096,
                "maxConcurrent": 1, "cpuThreads": threads})
            .to_string(),
        )
        .unwrap();
        path
    }

    /// The fixture graph: softmax over mean(ids) * w + count * 0.01.
    fn expected(ids: &[i64]) -> Vec<f64> {
        let mean = ids.iter().sum::<i64>() as f64 / ids.len() as f64;
        let logits: Vec<f64> = LEVEL_WEIGHTS
            .iter()
            .map(|w| mean * w + ids.len() as f64 * 0.01)
            .collect();
        let top = logits.iter().cloned().fold(f64::MIN, f64::max);
        let exp: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.iter().map(|e| e / sum).collect()
    }

    fn close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-5, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn native_scorer_builds_the_pinned_input_and_runs_the_graph() {
        let dir = tempfile::tempdir().unwrap();
        for threads in [1, 2] {
            let scorer = TractLevelScorer::open(
                &runtime(dir.path(), &fixture_dir(), threads),
                &binding(FIXTURE_MANIFEST),
            )
            .unwrap();
            // Prefix [2,2,2,2,2,1], then "fix login" as [3,4], then the separator 1.
            let out = scorer.score("Fix login").unwrap();
            close(&out.probabilities, &expected(&[2, 2, 2, 2, 2, 1, 3, 4, 1]));
            assert!(!out.truncated);
            // The mask token is removed before tokenizing.
            close(
                &scorer.score("fix [MASK] login").unwrap().probabilities,
                &out.probabilities,
            );
            // Only the room the input specification leaves is read.
            let long = scorer.score("fix login add tests across every").unwrap();
            assert!(long.truncated);
            close(
                &long.probabilities,
                &expected(&[2, 2, 2, 2, 2, 1, 3, 4, 5, 6, 1]),
            );
            assert!(scorer.score("   ").is_err());
            assert!(scorer.score(&"x".repeat(5000)).is_err());
        }
    }

    #[test]
    fn native_scorer_refuses_unpinned_or_altered_files() {
        let dir = tempfile::tempdir().unwrap();
        let config = runtime(dir.path(), &fixture_dir(), 1);
        let error = TractLevelScorer::open(&config, &binding(&"0".repeat(64)))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("manifest digest"), "{error}");
        let mut other = binding(FIXTURE_MANIFEST);
        other.id = "another-model".into();
        assert!(TractLevelScorer::open(&config, &other).is_err());
        // A tokenizer that differs from the manifest is refused.
        let copy = dir.path().join("copy");
        std::fs::create_dir(&copy).unwrap();
        for name in [
            "manifest.json",
            "model.onnx",
            "input.json",
            "tokenizer.json",
        ] {
            std::fs::copy(fixture_dir().join(name), copy.join(name)).unwrap();
        }
        let mut tokenizer = std::fs::read(copy.join("tokenizer.json")).unwrap();
        tokenizer.push(b'\n');
        std::fs::write(copy.join("tokenizer.json"), tokenizer).unwrap();
        let error =
            TractLevelScorer::open(&runtime(dir.path(), &copy, 1), &binding(FIXTURE_MANIFEST))
                .err()
                .unwrap()
                .to_string();
        assert!(error.contains("tokenizer digest"), "{error}");
        let mut bad: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
        bad["cpuThreads"] = serde_json::json!(0);
        std::fs::write(&config, bad.to_string()).unwrap();
        assert!(TractLevelScorer::open(&config, &binding(FIXTURE_MANIFEST)).is_err());
    }
}

#[cfg(test)]
mod load_tests {
    #[test]
    fn a_bound_model_needs_a_runtime_and_its_configuration() {
        let binding = lfg_core::assessment::ModelScoreBinding {
            id: "level-score-fixture".into(),
            manifest_sha256: "a".repeat(64),
            level_brackets: vec!["trivial".into(), "simple".into()],
            level_cut_points: vec![0.5],
        };
        if std::env::var_os("LFG_MODEL_SCORE_CONFIG").is_none() {
            let error = super::load(&binding).err().unwrap();
            if cfg!(feature = "onnx-shadow") {
                assert!(error.contains("LFG_MODEL_SCORE_CONFIG"), "{error}");
            } else {
                assert!(error.contains("onnx-shadow"), "{error}");
            }
        }
    }
}
