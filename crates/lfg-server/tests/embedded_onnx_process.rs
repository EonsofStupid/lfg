#![cfg(feature = "embedded-onnx")]

use std::{
    fs,
    io::Write,
    process::{Command, Stdio},
};

use serde_json::{Value, json};

fn predict(exe: &std::path::Path, cwd: &std::path::Path, args: &[&str]) -> std::process::Output {
    predict_text(exe, cwd, args, "Review the requested change")
}

fn predict_text(
    exe: &std::path::Path,
    cwd: &std::path::Path,
    args: &[&str],
    text: &str,
) -> std::process::Output {
    let mut child = Command::new(exe)
        .current_dir(cwd)
        .args(args)
        .env_remove("LFG_OPERATIONS_CONFIG")
        .env_remove("LFG_PREFLIGHT_CONFIG")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(format!("{}\n", json!({"id": "fixture", "text": text})).as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn copied_binary_runs_embedded_model_and_accepts_runtime_override() {
    let temp = tempfile::tempdir().unwrap();
    let exe = temp
        .path()
        .join(format!("lfg-server{}", std::env::consts::EXE_SUFFIX));
    fs::copy(env!("CARGO_BIN_EXE_lfg-server"), &exe).unwrap();
    let embedded = predict(&exe, temp.path(), &["embedded-onnx-predict"]);
    assert!(
        embedded.status.success(),
        "{}",
        String::from_utf8_lossy(&embedded.stderr)
    );
    let actual: Value = serde_json::from_slice(&embedded.stdout).unwrap();
    assert_eq!(actual["tokenization"]["truncated"], false);
    assert_eq!(actual["tokenization"]["overflow_chunks"], 0);

    let long = predict_text(
        &exe,
        temp.path(),
        &["embedded-onnx-predict"],
        &"review ".repeat(1000),
    );
    assert!(
        long.status.success(),
        "{}",
        String::from_utf8_lossy(&long.stderr)
    );
    let long_value: Value = serde_json::from_slice(&long.stdout).unwrap();
    assert_eq!(long_value["tokenization"]["truncated"], true);
    assert!(
        long_value["tokenization"]["overflow_chunks"]
            .as_u64()
            .unwrap()
            > 0
    );

    let artifact =
        std::env::var("LFG_EMBEDDED_ONNX_DIR").expect("feature build artifact directory");
    let digest = std::env::var("LFG_EMBEDDED_ONNX_SHA256").expect("feature build manifest digest");
    let config =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/onnx-runtime.example.json");
    let external = predict(
        &exe,
        temp.path(),
        &[
            "onnx-predict",
            "--artifact",
            &artifact,
            "--manifest-sha256",
            &digest,
            "--runtime-config",
            config.to_str().unwrap(),
        ],
    );
    assert!(
        external.status.success(),
        "{}",
        String::from_utf8_lossy(&external.stderr)
    );
    let expected: Value = serde_json::from_slice(&external.stdout).unwrap();
    assert_eq!(actual, expected);
    assert!(actual.get("scores").is_none());

    let scored = predict(
        &exe,
        temp.path(),
        &["embedded-onnx-predict", "--include-scores"],
    );
    assert!(
        scored.status.success(),
        "{}",
        String::from_utf8_lossy(&scored.stderr)
    );
    let mut scored_value: Value = serde_json::from_slice(&scored.stdout).unwrap();
    let scores = scored_value
        .as_object_mut()
        .unwrap()
        .remove("scores")
        .unwrap();
    assert_eq!(scored_value, actual);
    assert_eq!(scores["semantics"], "sigmoid-uncalibrated-v1");
    let values = scores["values"].as_object().unwrap();
    assert!(!values.is_empty());
    assert!(
        values
            .values()
            .all(|value| value.as_f64().is_some_and(|v| (0.0..=1.0).contains(&v)))
    );

    let scored_external = predict(
        &exe,
        temp.path(),
        &[
            "onnx-predict",
            "--artifact",
            &artifact,
            "--manifest-sha256",
            &digest,
            "--runtime-config",
            config.to_str().unwrap(),
            "--include-scores",
        ],
    );
    assert!(
        scored_external.status.success(),
        "{}",
        String::from_utf8_lossy(&scored_external.stderr)
    );
    let external_value: Value = serde_json::from_slice(&scored_external.stdout).unwrap();
    assert_eq!(external_value["scores"], scores);

    let mut limit: Value =
        serde_json::from_str(include_str!("../config/onnx-runtime.example.json")).unwrap();
    limit["maxSequenceTokens"] = json!(1);
    let config_path = temp.path().join("limit.json");
    fs::write(&config_path, serde_json::to_vec(&limit).unwrap()).unwrap();
    let rejected = predict(
        &exe,
        temp.path(),
        &[
            "embedded-onnx-predict",
            "--runtime-config",
            config_path.to_str().unwrap(),
        ],
    );
    assert!(!rejected.status.success());
}
