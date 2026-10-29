use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

struct RunningServer(Child);

impl Drop for RunningServer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn http_json(port: u16, method: &str, path: &str, body: Option<Value>) -> Value {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (headers, payload) = response.split_once("\r\n\r\n").unwrap();
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    serde_json::from_str(payload).unwrap()
}

#[test]
fn server_uses_external_configuration_and_rejects_invalid_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let mut classifier: Value =
        serde_json::from_str(include_str!("../../lfg-core/config/classifier.v2.json")).unwrap();
    classifier["project"] = json!("project-proof");
    classifier["revision"] = json!("profile-proof");
    classifier["escalationPercentThreshold"] = json!(0);
    classifier["escalationCatalog"] = json!([{"provider":"fixture","model":"fixture-model","role":"test","rawCapability":0.7,"toolLoopEfficiency":0.6,"costTier":"low","subscriptionId":"fixture-seat","strengths":["fixture"],"effortLevels":["test"],"evidence":{"source":"fixture","basis":"synthetic","observedAt":"test"}}]);
    let mut rules: Value =
        serde_json::from_str(include_str!("../../lfg-core/config/preflight.v1.json")).unwrap();
    rules["revision"] = json!("rules-proof");
    rules["rules"][0]["keywords"] = json!(["special-visual"]);
    let inventory = json!({"schemaVersion":"lfg.provider-inventory.v1","revision":"empty-proof","providers":[]});
    let mut rubric: Value = serde_json::from_str(include_str!(
        "../../lfg-core/config/evaluation-rubric.v1.json"
    ))
    .unwrap();
    rubric["passThreshold"] = json!(0.95);
    let mut policy: Value = serde_json::from_str(include_str!(
        "../../lfg-core/config/promotion-policy.v1.json"
    ))
    .unwrap();
    policy["minimumSamplesPerArm"] = json!(7);
    for (name, value) in [
        ("classifier.json", &classifier),
        ("preflight.json", &rules),
        ("providers.json", &inventory),
        ("rubric.json", &rubric),
        ("policy.json", &policy),
    ] {
        write_json(&root.join(name), value);
    }
    let port = TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
    command
        .args(["serve", "--bind", &format!("127.0.0.1:{port}")])
        .env("LFG_STATE_DIR", root.join("state"))
        .env("LFG_CLASSIFIER_CONFIG", root.join("classifier.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .env("LFG_PROVIDER_CONFIG", root.join("providers.json"))
        .env("LFG_EVALUATION_RUBRIC", root.join("rubric.json"))
        .env("LFG_PROMOTION_POLICY", root.join("policy.json"))
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut server = RunningServer(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited during startup"
        );
        assert!(Instant::now() < deadline, "server did not become ready");
        thread::sleep(Duration::from_millis(20));
    }
    let classified = http_json(
        port,
        "POST",
        "/api/lfg/classify",
        Some(
            json!({"sessionId":"test","revision":1,"text":"special-visual","activeSubscriptions":["fixture-seat"]}),
        ),
    );
    assert_eq!(classified["project"], "project-proof");
    assert_eq!(classified["configRevision"], "profile-proof");
    assert_eq!(
        classified["recommendedEscalation"]["model"],
        "fixture-model"
    );
    let preflight = http_json(
        port,
        "POST",
        "/api/shippin/intelligence/lfg/preflight",
        Some(json!({"sessionId":"test","revision":1,"text":"special-visual"})),
    );
    assert_eq!(preflight["model"]["version"], "rules-proof");
    assert_eq!(preflight["intents"][0]["label"], "create.image");
    assert!(preflight["confidence"].is_null());
    let status = http_json(port, "GET", "/api/shippin/intelligence/lfg/status", None);
    assert_eq!(status["configuredProviders"], 0);
    assert_eq!(status["fixtureProviders"], 0);
    let contracts = http_json(
        port,
        "GET",
        "/api/shippin/intelligence/lfg/evaluations/rubrics",
        None,
    );
    assert_eq!(contracts["rubrics"][0]["passThreshold"], 0.95);
    assert_eq!(contracts["promotionPolicy"]["minimumSamplesPerArm"], 7);
    drop(server);

    let output = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .arg("serve")
        .env("LFG_CLASSIFIER_CONFIG", root.join("missing.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .env("LFG_PROVIDER_CONFIG", root.join("providers.json"))
        .env("LFG_EVALUATION_RUBRIC", root.join("rubric.json"))
        .env("LFG_PROMOTION_POLICY", root.join("policy.json"))
        .env("LFG_STATE_DIR", root.join("invalid-state"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("classifier configuration"));
    assert!(!root.join("invalid-state").exists());
}
