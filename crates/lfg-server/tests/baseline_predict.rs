use std::io::Write;
use std::process::{Command, Stdio};

fn predict(input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .arg("baseline-predict")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn baseline_produces_metadata_without_prompt_content() {
    let input = "{\"id\":\"one\",\"text\":\"Implement the secret project and run tests\"}\n";
    let result = predict(input);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    assert!(!output.contains("secret project"));
    let prediction: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(prediction["schema_version"], "lfg.evaluation-prediction.v1");
    assert_eq!(prediction["id"], "one");
    assert_eq!(
        prediction["input_sha256"],
        lfg_core::digest_text("Implement the secret project and run tests")
    );
    assert_eq!(prediction["producer_sha256"].as_str().unwrap().len(), 64);
    assert!(
        prediction["intents"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "build.software")
    );
    assert!(
        prediction["risks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "sensitive-data")
    );
    assert!(prediction["model"]["sha256"].as_str().is_some());
}

#[test]
fn baseline_rejects_duplicate_ids() {
    let result =
        predict("{\"id\":\"same\",\"text\":\"one\"}\n{\"id\":\"same\",\"text\":\"two\"}\n");
    assert!(!result.status.success());
}
