use std::{env, net::SocketAddr, path::PathBuf};

use axum::Router;
use clap::{Parser, Subcommand};
use lfg_server::{AppState, router_with_ui};
use tokio::net::TcpListener;

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
    LfgBench {
        #[arg(long, default_value_t = 1000)]
        iterations: usize,
    },
    Serve {
        #[arg(long, default_value = "127.0.0.1:4317")]
        bind: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::LfgBench { iterations } => lfg_bench(iterations)?,
        Command::Serve { bind } => serve(bind).await?,
    }
    Ok(())
}

fn lfg_bench(iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    use lfg_core::{CpuRuleEncoder, PreflightEncoder, PreflightRequest};
    if iterations == 0 || iterations > 1_000_000 {
        return Err("iterations must be between 1 and 1000000".into());
    }
    let encoder = CpuRuleEncoder;
    let prompt = "Research current routing patterns, implement an image workflow, test it, and independently grade the result without exposing secrets.";
    let mut samples = Vec::with_capacity(iterations);
    for revision in 0..iterations {
        samples.push(
            encoder
                .encode(&PreflightRequest {
                    session_id: "benchmark".into(),
                    revision: revision as u64,
                    text: prompt.into(),
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

async fn serve(bind: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let state_root = env::var_os("LFG_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::temp_dir().join("lfg"));
    let ui_directory = env::var_os("MASTER_CLYFFY_UI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ui/master-clyffy-web/dist")
        });
    let app: Router = router_with_ui(
        AppState::persistent(state_root.join("signals.jsonl"))?,
        ui_directory,
    );
    let listener = TcpListener::bind(bind).await?;
    println!("lfg-server listening on http://{bind}");
    axum::serve(listener, app).await?;
    Ok(())
}
