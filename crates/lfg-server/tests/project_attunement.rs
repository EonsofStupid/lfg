use serde_json::{Value, json};
use std::{fs, process::Command};

fn invoke(root: &std::path::Path, config: &Value) -> std::process::Output {
    let config_path = root.join("adapter.json");
    fs::write(&config_path, config.to_string()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["boundary-project", "--root"])
        .arg(root)
        .arg("--config")
        .arg(config_path)
        .output()
        .unwrap()
}

fn config() -> Value {
    json!({"schemaVersion":"lfg.project-attunement.v1", "revision":"test-v1", "sourceApp":"test-project",
    "selections":[{"bankId":"core"},{"bankId":"typescript"}],
    "limits":{"maxEntries":20,"maxDepth":2,"maxFileBytes":512},
    "sources":[
        {"kind":"inventory","binding":"module inventory","path":"src","recursive":true,"extensions":["rs"]},
        {"kind":"json","binding":"tsconfig","path":"config.json","pointer":"/compilerOptions"}
    ]})
}

#[test]
fn project_evidence_fills_authored_bindings_without_answering_questions() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("src/nested")).unwrap();
    fs::write(
        root.path().join("src/nested/auth.rs"),
        "private contents not collected",
    )
    .unwrap();
    fs::write(root.path().join("src/ignore.txt"), "not included").unwrap();
    fs::write(
        root.path().join("config.json"),
        r#"{"compilerOptions":{"strict":true},"notSelected":"private"}"#,
    )
    .unwrap();
    let output = invoke(root.path(), &config());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    let questions = result["preview"]["questions"].as_array().unwrap();
    let by_binding = |binding: &str| {
        questions
            .iter()
            .find(|q| q["question"]["binding"] == binding)
            .unwrap()
    };
    assert_eq!(
        by_binding("module inventory")["artifact"],
        json!(["src/nested/auth.rs"])
    );
    assert_eq!(by_binding("tsconfig")["artifact"], json!({"strict":true}));
    assert_eq!(by_binding("boundary map")["bindingStatus"], "missing");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stdout.contains("private contents"));
    assert!(!stdout.contains("notSelected"));
    assert_eq!(result["sources"][0]["fileCount"], 1);
    assert_eq!(result["sources"][0]["recursive"], true);
    assert_eq!(result["sources"][0]["extensions"], json!(["rs"]));
    assert_eq!(result["sources"][1]["sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn rejects_invalid_or_unbounded_inputs_and_keeps_missing_source_missing() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(root.path().join("src/a.rs"), "").unwrap();
    let mut cfg = config();
    let output = invoke(root.path(), &cfg);
    assert!(output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["sources"][1]["status"], "missing");
    assert!(
        result["preview"]["missingBindings"]
            .as_array()
            .unwrap()
            .contains(&json!("tsconfig"))
    );
    cfg["sources"][0]["path"] = json!("../outside");
    assert!(!invoke(root.path(), &cfg).status.success());
    cfg = config();
    cfg["sources"][0]["binding"] = json!("invented binding");
    assert!(!invoke(root.path(), &cfg).status.success());
    cfg = config();
    cfg["limits"]["maxEntries"] = json!(1);
    fs::write(root.path().join("src/b.rs"), "").unwrap();
    assert!(!invoke(root.path(), &cfg).status.success());
    cfg = config();
    cfg["limits"]["maxFileBytes"] = json!(4);
    fs::write(root.path().join("config.json"), r#"{"compilerOptions":{}}"#).unwrap();
    assert!(!invoke(root.path(), &cfg).status.success());
    cfg = config();
    fs::write(
        root.path().join("config.json"),
        r#"{"compilerOptions":{"api_key":"secret"}}"#,
    )
    .unwrap();
    assert!(!invoke(root.path(), &cfg).status.success());
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_in_inventories_and_source_paths() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    symlink(outside.path(), root.path().join("src/link")).unwrap();
    assert!(!invoke(root.path(), &config()).status.success());
    let mut cfg = config();
    cfg["sources"][0]["path"] = json!("src/link");
    assert!(!invoke(root.path(), &cfg).status.success());
}
