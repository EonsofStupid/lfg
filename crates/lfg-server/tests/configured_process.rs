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

fn http_response(port: u16, method: &str, path: &str, body: Option<Value>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let body = body.map(|value| value.to_string()).unwrap_or_default();
    write!(stream, "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (headers, payload) = response.split_once("\r\n\r\n").unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, payload.to_owned())
}

fn http_json(port: u16, method: &str, path: &str, body: Option<Value>) -> Value {
    let (status, payload) = http_response(port, method, path, body);
    assert_eq!(status, 200);
    serde_json::from_str(&payload).unwrap()
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
    let inventory = json!({"schemaVersion":"lfg.provider-inventory.v1","revision":"fixture-proof","providers":[{"targetId":"fixture","displayName":"Fixture","provider":"test","adapterId":"fixture.adapter","model":"fixture-model","provenance":"fixture","availability":"replay-only","capabilities":["image.generate"]}]});
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
    let mut operations: Value =
        serde_json::from_str(include_str!("../../lfg-runtime/config/operations.v1.json")).unwrap();
    let mut intake: Value =
        serde_json::from_str(include_str!("../config/intake-template.v1.json")).unwrap();
    let seam_listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let seam_address = seam_listener.local_addr().unwrap();
    let seam = thread::spawn(move || {
        let (mut stream, _) = seam_listener.accept().unwrap();
        let mut request = [0_u8; 2048];
        let received = stream.read(&mut request).unwrap();
        assert!(String::from_utf8_lossy(&request[..received]).starts_with("GET /state "));
        let body = json!({"registry":{"providers":[{"id":"host-provider","activated":true,
            "capabilities":["read-sessions"],"note":"history only"}]}})
        .to_string();
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
    });
    let context = json!({"schemaVersion":"lfg.devpulse-context.v1","revision":"host-proof",
        "endpoint":seam_address.to_string(),"timeoutMs":1000,"pollIntervalMs":100,"maxResponseBytes":2048});
    intake["revision"] = json!("intake-proof");
    intake["questions"] = json!([{"id":"custom", "question":"What is requested?", "sourcePath":"/request/text", "kind":"string", "required":true}]);
    operations["revision"] = json!("operations-proof");
    operations["maxPromptBytes"] = json!(16);
    operations["maxHttpBodyBytes"] = json!(400);
    operations["defaultSignalPageSize"] = json!(1);
    operations["maxSignalPageSize"] = json!(2);
    operations["uiDebounceMs"] = json!(25);
    operations["devTools"]["traceCapacity"] = json!(2);
    operations["benchmarkDefaultIterations"] = json!(3);
    operations["benchmarkMaxIterations"] = json!(5);
    let port = TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    operations["defaultListenAddress"] = json!(format!("127.0.0.1:{port}"));
    for (name, value) in [
        ("classifier.json", &classifier),
        ("preflight.json", &rules),
        ("providers.json", &inventory),
        ("rubric.json", &rubric),
        ("policy.json", &policy),
        ("operations.json", &operations),
        ("intake.json", &intake),
        ("context.json", &context),
    ] {
        write_json(&root.join(name), value);
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
    command
        .arg("serve")
        .env("LFG_STATE_DIR", root.join("state"))
        .env("LFG_CLASSIFIER_CONFIG", root.join("classifier.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .env("LFG_PROVIDER_CONFIG", root.join("providers.json"))
        .env("LFG_EVALUATION_RUBRIC", root.join("rubric.json"))
        .env("LFG_PROMOTION_POLICY", root.join("policy.json"))
        .env("LFG_OPERATIONS_CONFIG", root.join("operations.json"))
        .env("LFG_INTAKE_TEMPLATE", root.join("intake.json"))
        .env("LFG_DEVPULSE_CONTEXT_CONFIG", root.join("context.json"))
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
    let preview = http_json(
        port,
        "POST",
        "/api/lfg/intake/preview",
        Some(
            json!({"schemaVersion":"lfg.app-snapshot.v1", "sourceApp":"host", "facts":{"request":{"text":"actual request"}}}),
        ),
    );
    assert_eq!(preview["templateRevision"], "intake-proof");
    assert_eq!(preview["answers"][0]["value"], "actual request");
    let context = http_json(port, "GET", "/api/lfg/intake/app-context", None);
    assert_eq!(context["status"], "ready");
    assert_eq!(context["providers"][0]["id"], "host-provider");
    assert_eq!(context["providers"][0]["capabilities"][0], "read-sessions");
    seam.join().unwrap();
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
    assert_eq!(status["fixtureProviders"], 1);
    assert_eq!(status["operations"]["revision"], "operations-proof");
    assert_eq!(status["operations"]["maxPromptBytes"], 16);
    assert_eq!(status["operations"]["uiDebounceMs"], 25);
    assert_eq!(status["operationsDigest"].as_str().unwrap().len(), 64);
    let oversized = Some(json!({"sessionId":"test","revision":1,"text":"ééééééééé"}));
    assert_eq!(
        http_response(port, "POST", "/api/lfg/classify", oversized.clone()).0,
        400
    );
    assert_eq!(
        http_response(
            port,
            "POST",
            "/api/shippin/intelligence/lfg/preflight",
            oversized.clone()
        )
        .0,
        400
    );
    assert_eq!(
        http_response(
            port,
            "POST",
            "/api/shippin/intelligence/lfg/runs",
            oversized
        )
        .0,
        400
    );
    assert_eq!(
        http_response(
            port,
            "POST",
            "/api/lfg/classify",
            Some(json!({"sessionId":"test","revision":1,"text":"x".repeat(500)}))
        )
        .0,
        413
    );
    let stats = http_json(port, "GET", "/api/lfg/dev-stats", None);
    assert_eq!(stats["enabled"], true);
    assert_eq!(stats["capacity"], 2);
    assert_eq!(stats["totalRecorded"], 4);
    assert_eq!(stats["evicted"], 2);
    assert!(stats["cpuTimeMicros"].is_null());
    assert_eq!(stats["operationsDigest"], status["operationsDigest"]);
    assert_eq!(stats["encoderDigest"], status["encoder"]["sha256"]);
    assert!(!stats.to_string().contains("special-visual"));
    assert_eq!(stats["traces"].as_array().unwrap().len(), 2);
    let (code, plan_body) = http_response(
        port,
        "POST",
        "/api/shippin/intelligence/lfg/plans",
        Some(
            json!({"sessionId":"test","revision":1,"text":"special-visual","includeFixtureBindings":true}),
        ),
    );
    assert_eq!(code, 201);
    let plan: Value = serde_json::from_str(&plan_body).unwrap();
    let workflow = plan["workflowId"].as_str().unwrap();
    let page = http_json(
        port,
        "GET",
        &format!("/api/shippin/intelligence/lfg/signals?workflowId={workflow}"),
        None,
    );
    assert_eq!(page["signals"].as_array().unwrap().len(), 1);
    let page = http_json(
        port,
        "GET",
        &format!("/api/shippin/intelligence/lfg/signals?workflowId={workflow}&limit=99"),
        None,
    );
    assert_eq!(page["signals"].as_array().unwrap().len(), 2);
    let replay = http_json(
        port,
        "GET",
        &format!("/api/shippin/intelligence/lfg/signals/replay?workflowId={workflow}&limit=99"),
        None,
    );
    assert_eq!(replay["signals"].as_array().unwrap().len(), 2);
    let contracts = http_json(
        port,
        "GET",
        "/api/shippin/intelligence/lfg/evaluations/rubrics",
        None,
    );
    assert_eq!(contracts["rubrics"][0]["passThreshold"], 0.95);
    assert_eq!(contracts["promotionPolicy"]["minimumSamplesPerArm"], 7);
    drop(server);

    fs::write(root.join("benchmark.txt"), "special-visual").unwrap();
    let benchmark = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["lfg-bench", "--prompt-file"])
        .arg(root.join("benchmark.txt"))
        .env("LFG_OPERATIONS_CONFIG", root.join("operations.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .output()
        .unwrap();
    assert!(benchmark.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&benchmark.stdout).unwrap()["iterations"],
        3
    );
    let oversized_iterations = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["lfg-bench", "--iterations", "6", "--prompt-file"])
        .arg(root.join("benchmark.txt"))
        .env("LFG_OPERATIONS_CONFIG", root.join("operations.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .output()
        .unwrap();
    assert!(!oversized_iterations.status.success());
    fs::write(root.join("oversized-benchmark.txt"), "é".repeat(9)).unwrap();
    let oversized_prompt = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["lfg-bench", "--prompt-file"])
        .arg(root.join("oversized-benchmark.txt"))
        .env("LFG_OPERATIONS_CONFIG", root.join("operations.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .output()
        .unwrap();
    assert!(!oversized_prompt.status.success());
    assert!(String::from_utf8_lossy(&oversized_prompt.stderr).contains("exceeds 16 UTF-8 bytes"));

    let output = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .arg("serve")
        .env("LFG_CLASSIFIER_CONFIG", root.join("missing.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .env("LFG_PROVIDER_CONFIG", root.join("providers.json"))
        .env("LFG_EVALUATION_RUBRIC", root.join("rubric.json"))
        .env("LFG_PROMOTION_POLICY", root.join("policy.json"))
        .env("LFG_OPERATIONS_CONFIG", root.join("operations.json"))
        .env("LFG_STATE_DIR", root.join("invalid-state"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("classifier configuration"));
    assert!(!root.join("invalid-state").exists());

    operations["eventBufferCapacity"] = json!(0);
    write_json(&root.join("invalid-operations.json"), &operations);
    let output = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .arg("serve")
        .env("LFG_CLASSIFIER_CONFIG", root.join("classifier.json"))
        .env("LFG_PREFLIGHT_CONFIG", root.join("preflight.json"))
        .env("LFG_PROVIDER_CONFIG", root.join("providers.json"))
        .env("LFG_EVALUATION_RUBRIC", root.join("rubric.json"))
        .env("LFG_PROMOTION_POLICY", root.join("policy.json"))
        .env(
            "LFG_OPERATIONS_CONFIG",
            root.join("invalid-operations.json"),
        )
        .env("LFG_STATE_DIR", root.join("invalid-operations-state"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("operational configuration"));
    assert!(!root.join("invalid-operations-state").exists());
}
