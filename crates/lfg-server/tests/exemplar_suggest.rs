#![cfg(feature = "onnx-shadow")]
//! Mechanical fixtures only; simulated admission is not training evidence.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn digest(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

#[test]
fn resolved_kernel_paths_do_not_load_the_encoder_and_escalations_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.json");
    let runtime = dir.path().join("runtime.json");
    let artifact = dir.path().join("artifact");
    std::fs::create_dir(&artifact).unwrap();
    let config_raw = include_bytes!("../../../ml/lfg-encoder/node-exemplar-config.example.json");
    std::fs::write(&config_path, config_raw).unwrap();
    std::fs::write(
        &runtime,
        include_bytes!("../config/node-onnx-runtime.example.json"),
    )
    .unwrap();
    let config: Value = serde_json::from_slice(config_raw).unwrap();
    let encoder_sha = lfg_core::digest_value(
        &json!({"pooling":config["pooling"],"maxTokens":config["maxTokens"],
        "maxTextBytes":config["maxTextBytes"],"batchSize":config["batchSize"],"cpuThreads":config["cpuThreads"]}),
    );
    let manifest = json!({"schemaVersion":"lfg.node-onnx-encoder.v1", "promotionEligible":false,
        "modelSha256":"a".repeat(64),"tokenizerSha256":"b".repeat(64),"sourceModelManifestSha256":"c".repeat(64),
        "encoderConfigSha256":encoder_sha,"exportConfigSha256":"d".repeat(64),
        "inputs":["input_ids","attention_mask"],"output":"embedding","dimension":2,"maxTokens":256,
        "pooling":"attention-mask-mean-l2"});
    let manifest_raw = serde_json::to_vec(&manifest).unwrap();
    std::fs::write(artifact.join("manifest.json"), &manifest_raw).unwrap();
    let engine = lfg_core::kernel::Engine::bundled().unwrap();
    let pack = json!({"schemaVersion":"lfg.node-exemplar-pack.v1", "promotionEligible":false,
        "node":"operation","snapshotSha256":engine.snapshot_digest(),
        "admission":{"schemaVersion":"lfg.node-admission-report.v1","admissionEligible":true,
            "qualificationEstablished":false,"node":"operation","snapshotSha256":engine.snapshot_digest(),
            "pending":0,"approved":6,"supportGatePassed":true,"projectGroupGatePassed":true,
            "conflicts":[],"counts":{"train":{"chat":1,"tool_call":1}},
            "sources":[{"queueSha256":"1".repeat(64),"ledgerSha256":"2".repeat(64)}]},
        "model":{"manifest_sha256":"c".repeat(64),"model_id":"fixture","revision":"fixture","files":1},
        "environment":{"lock_sha256":"3".repeat(64),"installed_versions_sha256":"4".repeat(64),
            "locked_wheel_payload":{"schema_version":"lfg.installed-wheel-payload.v1","payload_sha256":"5".repeat(64),"files":1}},
        "encoderVersions":{"torch":"fixture","transformers":"fixture"},
        "buildConfigSha256":"d".repeat(64),"encoderConfigSha256":encoder_sha,
        "options":["chat","tool_call"],"dimension":2,
        "exemplars":[{"itemSha256":"a".repeat(64),"choice":"chat","embedding":[1.0,0.0]},
                     {"itemSha256":"b".repeat(64),"choice":"tool_call","embedding":[0.0,1.0]}],
        "builderSha256":"e".repeat(64),"inputVerifierSha256":"f".repeat(64)});
    let pack_raw = serde_json::to_vec(&pack).unwrap();
    let pack_path = dir.path().join("pack.json");
    std::fs::write(&pack_path, &pack_raw).unwrap();
    let invoke = |text: &str, experimental: bool, prediction_only: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
        command
            .args(["onnx-node-suggest", "--artifact"])
            .arg(&artifact)
            .args([
                "--manifest-sha256",
                &digest(&manifest_raw),
                "--runtime-config",
            ])
            .arg(&runtime)
            .arg("--exemplar-pack")
            .arg(&pack_path)
            .args(["--pack-sha256", &digest(&pack_raw), "--exemplar-config"])
            .arg(&config_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if experimental {
            command.arg("--experimental");
        }
        if prediction_only {
            command.arg("--prediction-only");
        }
        let mut child = command.spawn().unwrap();
        writeln!(
            child.stdin.take().unwrap(),
            "{}",
            json!({"id":"fixture","text":text})
        )
        .unwrap();
        child.wait_with_output().unwrap()
    };
    assert!(!invoke("run command", false, false).status.success());
    let resolved = invoke("run command", true, false);
    assert!(
        resolved.status.success(),
        "{}",
        String::from_utf8_lossy(&resolved.stderr)
    );
    let row: Value = serde_json::from_slice(&resolved.stdout).unwrap();
    assert_eq!(row["extensionStatus"], "not_applicable");
    assert!(row["suggestion"].is_null());
    assert_eq!(row["promotionEligible"], false);
    let compact = invoke("run command", true, true);
    assert!(compact.status.success());
    let row: Value = serde_json::from_slice(&compact.stdout).unwrap();
    assert_eq!(row["schemaVersion"], "lfg.node-prediction.v1");
    assert!(row["choice"].is_null());
    let escalated = invoke("ordinary unmatched fixture", true, false);
    assert!(!escalated.status.success());
    assert!(escalated.stdout.is_empty());
}
