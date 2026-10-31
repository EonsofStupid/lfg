use std::process::Command;

#[test]
fn embedded_snapshot_classifies_the_authored_golden_and_escalates_unknown() {
    let run = |prompt| {
        Command::new(env!("CARGO_BIN_EXE_lfg-server"))
            .args(["classify", prompt])
            .output()
            .unwrap()
    };
    let result = run("logout doesn't clear the session");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let expected = include_bytes!("fixtures/kernel-logout.v1.json");
    assert_eq!(result.stdout, expected);
    assert_eq!(run("logout doesn't clear the session").stdout, expected);
    let unknown = run("unmatched material");
    assert!(unknown.status.success());
    let value: serde_json::Value = serde_json::from_slice(&unknown.stdout).unwrap();
    assert_eq!(value["state"], "escalate");
    assert_eq!(value["reason"], "no_matching_rule");
    assert_eq!(value["address"], "rr1:core@r1");
}

#[test]
fn external_snapshot_changes_rules_and_invalid_snapshots_fail_closed() {
    let mut config: serde_json::Value =
        serde_json::from_str(include_str!("../../lfg-core/config/kernel-core.v1.json")).unwrap();
    config["revision"] = serde_json::json!("r2");
    config["nodes"]["operation"]["rules"][0]["pattern"] = serde_json::json!("customword");
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("snapshot.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["classify", "customword session", "--snapshot"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    let result: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result["address"], "rr1:core/chat/code.fix/auth@r2");
    config["nodes"]["done"] = serde_json::json!({"kind":"gate","next":"operation","rules":[
        {"id":"cycle","field":"prompt","pattern":"x","open":true}]});
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["classify", "customword session", "--snapshot"])
        .arg(path)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
}
