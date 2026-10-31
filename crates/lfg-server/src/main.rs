use std::{collections::BTreeMap, env, io::Read, net::SocketAddr, path::PathBuf};

use axum::Router;
use clap::{Parser, Subcommand};
use lfg_server::{AppState, router_with_ui};
use tokio::net::TcpListener;

mod baseline_predict;
mod kernel_journal;
mod kernel_predict;
use kernel_journal::Journal;
#[cfg(feature = "onnx-shadow")]
mod onnx_predict;

#[derive(Debug, Parser)]
#[command(
    name = "lfg-server",
    about = "LFG prompt-recipe encoder + routing service"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Classify with the experimental deterministic LFG kernel.
    Classify {
        prompt: String,
        #[arg(long)]
        snapshot: Option<PathBuf>,
        /// Additional named input fields as NAME=VALUE.
        #[arg(long = "field")]
        fields: Vec<String>,
    },
    /// Stream real examples through one compiled kernel for offline analysis.
    KernelPredict {
        #[arg(long)]
        snapshot: Option<PathBuf>,
        #[arg(long)]
        max_examples: usize,
        #[arg(long)]
        journal: Option<PathBuf>,
        #[arg(long)]
        journal_max_bytes: Option<u64>,
        /// Explicitly retain input text in the local journal for later review.
        #[arg(long)]
        capture_text: bool,
    },
    /// Verify a local kernel journal and show aggregate counts.
    KernelJournalStatus {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        journal_max_bytes: u64,
    },
    /// Display one observation and valid snapshot choices for review.
    KernelJournalShow {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        journal_max_bytes: u64,
        #[arg(long)]
        observation: u64,
        #[arg(long)]
        snapshot: Option<PathBuf>,
        /// Display text from the journal or an explicitly selected, verified source.
        #[arg(long)]
        include_text: bool,
        #[arg(long)]
        source_jsonl: Option<PathBuf>,
        #[arg(long)]
        source_max_bytes: Option<u64>,
    },
    /// Match journal observation IDs and digests to an explicitly selected source file.
    KernelJournalReconcile {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        journal_max_bytes: u64,
        #[arg(long)]
        source_jsonl: PathBuf,
        #[arg(long)]
        source_max_bytes: u64,
    },
    /// Export corrected observations with available text as pending per-node review candidates.
    KernelJournalExport {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        journal_max_bytes: u64,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        output_max_bytes: u64,
        #[arg(long)]
        snapshot: Option<PathBuf>,
        #[arg(long)]
        source_jsonl: Option<PathBuf>,
        #[arg(long)]
        source_max_bytes: Option<u64>,
    },
    /// Append an explicitly attested correction to a prior kernel observation.
    KernelCorrect {
        #[arg(long)]
        journal: PathBuf,
        #[arg(long)]
        journal_max_bytes: u64,
        #[arg(long)]
        observation: u64,
        #[arg(long)]
        node: String,
        #[arg(long)]
        choice: String,
        #[arg(long)]
        reviewer: String,
        #[arg(long)]
        attest: bool,
        #[arg(long)]
        snapshot: Option<PathBuf>,
    },
    /// Read JSONL examples from stdin and emit metadata-only baseline predictions.
    BaselinePredict,
    /// Run an unpromoted ONNX artifact for offline comparison only.
    #[cfg(feature = "onnx-shadow")]
    OnnxPredict {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long)]
        manifest_sha256: String,
        #[arg(long)]
        runtime_config: PathBuf,
        /// Include uncalibrated per-label sigmoid scores for offline evaluation.
        #[arg(long)]
        include_scores: bool,
    },
    /// Run the binary's pinned, unpromoted embedded ONNX artifact for offline comparison.
    #[cfg(feature = "embedded-onnx")]
    EmbeddedOnnxPredict {
        /// Override the bundled runtime limits with a local configuration file.
        #[arg(long)]
        runtime_config: Option<PathBuf>,
        /// Include uncalibrated per-label sigmoid scores for offline evaluation.
        #[arg(long)]
        include_scores: bool,
    },
    LfgBench {
        #[arg(long)]
        iterations: Option<usize>,
        #[arg(long)]
        prompt_file: Option<PathBuf>,
    },
    Serve {
        #[arg(long)]
        bind: Option<SocketAddr>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Classify {
            prompt,
            snapshot,
            fields,
        } => classify(prompt, snapshot, fields)?,
        Command::KernelPredict {
            snapshot,
            max_examples,
            journal,
            journal_max_bytes,
            capture_text,
        } => kernel_predict::run(
            snapshot,
            max_examples,
            journal,
            journal_max_bytes,
            capture_text,
        )?,
        Command::KernelJournalStatus {
            journal,
            journal_max_bytes,
        } => {
            let store = kernel_journal::JsonlJournal::open(&journal, journal_max_bytes, false)?;
            println!("{}", store.status());
        }
        Command::KernelJournalShow {
            journal,
            journal_max_bytes,
            observation,
            snapshot,
            include_text,
            source_jsonl,
            source_max_bytes,
        } => {
            let engine = load_kernel(snapshot)?;
            let store = kernel_journal::JsonlJournal::open(&journal, journal_max_bytes, false)?;
            let source = load_source(source_jsonl, source_max_bytes)?;
            println!(
                "{}",
                store.show(&engine, observation, include_text, source.as_ref())?
            );
        }
        Command::KernelJournalReconcile {
            journal,
            journal_max_bytes,
            source_jsonl,
            source_max_bytes,
        } => {
            let store = kernel_journal::JsonlJournal::open(&journal, journal_max_bytes, false)?;
            let source = kernel_journal::SourceIndex::read(&source_jsonl, source_max_bytes)?;
            println!("{}", store.reconcile(&source)?);
        }
        Command::KernelJournalExport {
            journal,
            journal_max_bytes,
            output,
            output_max_bytes,
            snapshot,
            source_jsonl,
            source_max_bytes,
        } => {
            let engine = load_kernel(snapshot)?;
            let store = kernel_journal::JsonlJournal::open(&journal, journal_max_bytes, false)?;
            let source = load_source(source_jsonl, source_max_bytes)?;
            println!(
                "{}",
                store.export_candidates(&engine, &output, output_max_bytes, source.as_ref())?
            );
        }
        Command::KernelCorrect {
            journal,
            journal_max_bytes,
            observation,
            node,
            choice,
            reviewer,
            attest,
            snapshot,
        } => {
            if !attest {
                return Err("correction requires explicit --attest".into());
            }
            let engine = load_kernel(snapshot)?;
            let mut store = kernel_journal::JsonlJournal::open(&journal, journal_max_bytes, false)?;
            let sequence = store.correct(&engine, observation, &node, &choice, &reviewer)?;
            println!(
                "{}",
                serde_json::json!({"correctionSequence":sequence,"observationSequence":observation})
            );
        }
        Command::BaselinePredict => baseline_predict::run()?,
        #[cfg(feature = "onnx-shadow")]
        Command::OnnxPredict {
            artifact,
            manifest_sha256,
            runtime_config,
            include_scores,
        } => onnx_predict::run(&artifact, &manifest_sha256, &runtime_config, include_scores)?,
        #[cfg(feature = "embedded-onnx")]
        Command::EmbeddedOnnxPredict {
            runtime_config,
            include_scores,
        } => onnx_predict::run_embedded(runtime_config.as_deref(), include_scores)?,
        Command::LfgBench {
            iterations,
            prompt_file,
        } => lfg_bench(iterations, prompt_file)?,
        Command::Serve { bind } => serve(bind).await?,
    }
    Ok(())
}

fn load_source(
    path: Option<PathBuf>,
    limit: Option<u64>,
) -> Result<Option<kernel_journal::SourceIndex>, Box<dyn std::error::Error>> {
    match (path, limit) {
        (Some(path), Some(limit)) => Ok(Some(kernel_journal::SourceIndex::read(&path, limit)?)),
        (None, None) => Ok(None),
        _ => Err("source path and source byte limit must be supplied together".into()),
    }
}

fn classify(
    prompt: String,
    snapshot: Option<PathBuf>,
    additional: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut fields = BTreeMap::from([("prompt".to_string(), prompt)]);
    for field in additional {
        let (name, value) = field.split_once('=').ok_or("fields must be NAME=VALUE")?;
        if fields.insert(name.to_owned(), value.to_owned()).is_some() {
            return Err("duplicate input field".into());
        }
    }
    let engine = load_kernel(snapshot)?;
    println!("{}", serde_json::to_string(&engine.evaluate(&fields)?)?);
    Ok(())
}

fn load_kernel(
    snapshot: Option<PathBuf>,
) -> Result<lfg_core::kernel::Engine, Box<dyn std::error::Error>> {
    if let Some(path) = snapshot {
        let limit = lfg_server::configured_operations()?.max_http_body_bytes();
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take((limit as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() > limit {
            return Err("snapshot exceeds configured byte limit".into());
        }
        Ok(lfg_core::kernel::Engine::from_json(&bytes)?)
    } else {
        Ok(lfg_core::kernel::Engine::bundled()?)
    }
}

fn lfg_bench(
    iterations: Option<usize>,
    prompt_file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    use lfg_core::{PreflightEncoder, PreflightRequest};
    let operations = lfg_server::configured_operations()?;
    let iterations = iterations.unwrap_or(operations.benchmark_default_iterations());
    if iterations == 0 || iterations > operations.benchmark_max_iterations() {
        return Err(format!(
            "iterations must be between 1 and {}",
            operations.benchmark_max_iterations()
        )
        .into());
    }
    let encoder = lfg_server::configured_preflight_encoder()?;
    let prompt = match prompt_file {
        Some(path) => {
            let mut bytes = Vec::new();
            std::fs::File::open(path)?
                .take(operations.max_prompt_bytes() as u64 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > operations.max_prompt_bytes() {
                return Err(format!(
                    "benchmark prompt exceeds {} UTF-8 bytes",
                    operations.max_prompt_bytes()
                )
                .into());
            }
            String::from_utf8(bytes)?
        }
        None => include_str!("../config/benchmark-prompt.txt")
            .trim_end()
            .to_owned(),
    };
    if prompt.trim().is_empty() || prompt.len() > operations.max_prompt_bytes() {
        return Err(format!(
            "benchmark prompt must be nonempty and at most {} UTF-8 bytes",
            operations.max_prompt_bytes()
        )
        .into());
    }
    let mut samples = Vec::with_capacity(iterations);
    for revision in 0..iterations {
        samples.push(
            encoder
                .encode(&PreflightRequest {
                    session_id: "benchmark".into(),
                    revision: revision as u64,
                    text: prompt.clone(),
                })
                .duration_micros,
        );
    }
    samples.sort_unstable();
    let percentile = |value: f64| samples[((samples.len() - 1) as f64 * value).round() as usize];
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schemaVersion": "lfg.encoder-benchmark.v1",
            "encoder": encoder.manifest(),
            "iterations": iterations,
            "units": "microseconds",
            "min": samples[0],
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": samples[samples.len() - 1]
        }))?
    );
    Ok(())
}

async fn serve(bind: Option<SocketAddr>) -> Result<(), Box<dyn std::error::Error>> {
    let state_root = env::var_os("LFG_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("lfg"));
    let state = AppState::persistent(state_root.join("signals.jsonl"))?;
    let bind = bind.unwrap_or_else(|| state.operations().default_listen_address());
    let external_ui = env::var_os("MASTER_CLYFFY_UI_DIR").map(PathBuf::from);
    #[cfg(feature = "embedded-ui")]
    let app: Router = match external_ui {
        Some(directory) => router_with_ui(state, directory),
        None => lfg_server::router_with_embedded_ui(state),
    };
    #[cfg(not(feature = "embedded-ui"))]
    let app: Router = router_with_ui(
        state,
        external_ui.unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ui/master-clyffy-web/dist")
        }),
    );
    let listener = TcpListener::bind(bind).await?;
    println!("lfg-server listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
