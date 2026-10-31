#![cfg(feature = "embedded-node")]
//! Only for explicitly selected test artifacts; never a release or quality claim.

use serde_json::{Value, json};
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

fn invoke(binary: &Path, directory: &Path, args: &[&str], text: &str) -> std::process::Output {
    let mut child = Command::new(binary)
        .current_dir(directory)
        .args(args)
        .env_remove("LFG_OPERATIONS_CONFIG")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = writeln!(
        child.stdin.take().unwrap(),
        "{}",
        json!({"id":"fixture", "text":text})
    );
    child.wait_with_output().unwrap()
}

#[test]
fn copied_binary_runs_embedded_node_without_artifact_files() {
    let isolated = tempfile::tempdir().unwrap();
    let binary = isolated.path().join(if cfg!(windows) {
        "lfg-test.exe"
    } else {
        "lfg-test"
    });
    std::fs::copy(env!("CARGO_BIN_EXE_lfg-server"), &binary).unwrap();
    let no_flag = invoke(
        &binary,
        isolated.path(),
        &["embedded-node-suggest"],
        "fixture unknown",
    );
    assert!(!no_flag.status.success());
    let result = invoke(
        &binary,
        isolated.path(),
        &["embedded-node-suggest", "--experimental"],
        "fixture unknown",
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let row: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(row["baseline"]["state"], "escalate");
    assert_ne!(row["extensionStatus"], "not_applicable");
    assert_eq!(row["promotionEligible"], false);
    let artifact = std::env::var("LFG_EMBEDDED_NODE_DIR").unwrap();
    let manifest = std::env::var("LFG_EMBEDDED_NODE_MANIFEST_SHA256").unwrap();
    let pack = std::env::var("LFG_EMBEDDED_NODE_PACK").unwrap();
    let pack_sha = std::env::var("LFG_EMBEDDED_NODE_PACK_SHA256").unwrap();
    let snapshot = std::env::var("LFG_EMBEDDED_NODE_SNAPSHOT").unwrap();
    let runtime = std::env::var("LFG_EMBEDDED_NODE_RUNTIME").unwrap();
    let config = std::env::var("LFG_EMBEDDED_NODE_CONFIG").unwrap();
    let external = invoke(
        &binary,
        isolated.path(),
        &[
            "onnx-node-suggest",
            "--experimental",
            "--artifact",
            &artifact,
            "--manifest-sha256",
            &manifest,
            "--runtime-config",
            &runtime,
            "--exemplar-pack",
            &pack,
            "--pack-sha256",
            &pack_sha,
            "--exemplar-config",
            &config,
            "--snapshot",
            &snapshot,
        ],
        "fixture unknown",
    );
    assert!(
        external.status.success(),
        "{}",
        String::from_utf8_lossy(&external.stderr)
    );
    let other: Value = serde_json::from_slice(&external.stdout).unwrap();
    assert_eq!(row["baseline"], other["baseline"]);
    assert_eq!(row["extensionStatus"], other["extensionStatus"]);
    assert_eq!(row["suggestion"]["choice"], other["suggestion"]["choice"]);
    if !row["suggestion"].is_null() {
        for key in ["similarity", "margin"] {
            assert!(
                (row["suggestion"][key].as_f64().unwrap()
                    - other["suggestion"][key].as_f64().unwrap())
                .abs()
                    < 1e-6
            );
        }
    }
    let strict_path = isolated.path().join("strict-config.json");
    let mut strict: Value = serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    strict["minimumMargin"] = json!(2.0);
    std::fs::write(&strict_path, serde_json::to_vec(&strict).unwrap()).unwrap();
    let abstained = invoke(
        &binary,
        isolated.path(),
        &[
            "embedded-node-suggest",
            "--experimental",
            "--exemplar-config",
            strict_path.to_str().unwrap(),
        ],
        "fixture unknown",
    );
    assert!(
        abstained.status.success(),
        "{}",
        String::from_utf8_lossy(&abstained.stderr)
    );
    let abstained: Value = serde_json::from_slice(&abstained.stdout).unwrap();
    assert_eq!(abstained["extensionStatus"], "abstained");
    assert!(abstained["suggestion"].is_null());
    let compact = invoke(
        &binary,
        isolated.path(),
        &[
            "embedded-node-suggest",
            "--experimental",
            "--prediction-only",
        ],
        "run command",
    );
    assert!(compact.status.success());
    let compact: Value = serde_json::from_slice(&compact.stdout).unwrap();
    assert_eq!(compact["schemaVersion"], "lfg.node-prediction.v1");
    assert!(compact["choice"].is_null());
    let bad_config = isolated.path().join("invalid.json");
    std::fs::write(&bad_config, b"{}").unwrap();
    let bad = invoke(
        &binary,
        isolated.path(),
        &[
            "embedded-node-suggest",
            "--experimental",
            "--runtime-config",
            bad_config.to_str().unwrap(),
        ],
        "fixture unknown",
    );
    assert!(!bad.status.success());
    assert!(bad.stdout.is_empty());
}
