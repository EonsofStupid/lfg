#![cfg(feature = "owned-byte-onnx")]

use std::{fs, process::Command};

#[test]
fn probe_requires_opt_in_and_checks_manifest_before_loading_model() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("runtime.json");
    fs::write(
        &config,
        include_bytes!("../config/owned-byte-runtime.example.json"),
    )
    .unwrap();
    fs::write(temp.path().join("manifest.json"), b"{}").unwrap();
    let exe = env!("CARGO_BIN_EXE_lfg-server");
    let args = [
        "owned-byte-probe",
        "--artifact",
        temp.path().to_str().unwrap(),
        "--manifest-sha256",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "--runtime-config",
        config.to_str().unwrap(),
    ];
    let no_opt_in = Command::new(exe).args(args).output().unwrap();
    assert!(!no_opt_in.status.success());
    assert!(String::from_utf8_lossy(&no_opt_in.stderr).contains("requires --experimental"));
    let wrong_digest = Command::new(exe)
        .args(args)
        .arg("--experimental")
        .output()
        .unwrap();
    assert!(!wrong_digest.status.success());
    assert!(String::from_utf8_lossy(&wrong_digest.stderr).contains("manifest digest mismatch"));
}

#[cfg(feature = "embedded-owned-byte")]
#[test]
fn embedded_probe_runs_in_empty_directory_and_checks_context() {
    use std::{io::Write, process::Stdio};

    let temp = tempfile::tempdir().unwrap();
    let run = |request: serde_json::Value| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
            .args(["embedded-owned-byte-probe", "--experimental"])
            .current_dir(temp.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(child.stdin.take().unwrap(), "{request}").unwrap();
        child.wait_with_output().unwrap()
    };
    let good = run(serde_json::json!({"id":"fixture", "text":"abc", "extra": true}));
    assert!(
        good.status.success(),
        "{}",
        String::from_utf8_lossy(&good.stderr)
    );
    let row: serde_json::Value = serde_json::from_slice(&good.stdout).unwrap();
    assert_eq!(row["input_bytes"], 3);
    assert!(row.get("logits").is_none());
    let root = std::env::var_os("LFG_EMBEDDED_OWNED_BYTE_DIR").unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(std::path::Path::new(&root).join("manifest.json")).unwrap(),
    )
    .unwrap();
    let over = "x".repeat(manifest["context"].as_u64().unwrap() as usize + 1);
    let rejected = run(serde_json::json!({"id":"fixture", "text":over}));
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("context"));
}
