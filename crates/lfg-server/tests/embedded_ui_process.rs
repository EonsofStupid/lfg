#![cfg(feature = "embedded-ui")]

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn start(root: &Path, external_ui: Option<&Path>) -> (Server, u16) {
    let executable = root.join(format!("lfg-server{}", std::env::consts::EXE_SUFFIX));
    fs::copy(env!("CARGO_BIN_EXE_lfg-server"), &executable).unwrap();
    let port = TcpListener::bind(("127.0.0.1", 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut command = Command::new(executable);
    command
        .current_dir(root)
        .args(["serve", "--bind", &format!("127.0.0.1:{port}")])
        .env("LFG_STATE_DIR", root.join("state"))
        .env_remove("MASTER_CLYFFY_UI_DIR")
        .env_remove("LFG_OPERATIONS_CONFIG")
        .env_remove("LFG_PREFLIGHT_CONFIG")
        .env_remove("LFG_CLASSIFIER_CONFIG")
        .env_remove("LFG_PROVIDER_CONFIG")
        .env_remove("LFG_EVALUATION_RUBRIC")
        .env_remove("LFG_PROMOTION_POLICY")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(path) = external_ui {
        command.env("MASTER_CLYFFY_UI_DIR", path);
    }
    let mut server = Server(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "server exited during startup"
        );
        assert!(Instant::now() < deadline, "server did not become ready");
        thread::sleep(Duration::from_millis(20));
    }
    (server, port)
}

fn get(port: u16, method: &str, path: &str) -> (u16, String, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let index = response
        .windows(4)
        .position(|part| part == b"\r\n\r\n")
        .unwrap();
    let headers = String::from_utf8(response[..index].to_vec()).unwrap();
    let status = headers.split_whitespace().nth(1).unwrap().parse().unwrap();
    (
        status,
        headers.to_lowercase(),
        response[index + 4..].to_vec(),
    )
}

#[test]
fn copied_binary_serves_ui_and_api_without_checkout() {
    let temp = tempfile::tempdir().unwrap();
    let (_server, port) = start(temp.path(), None);
    let (status, headers, html) = get(port, "GET", "/");
    assert_eq!(status, 200);
    assert!(headers.contains("content-type: text/html"));
    let html = String::from_utf8(html).unwrap();
    let asset = html
        .split("src=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    let (status, headers, javascript) = get(port, "GET", asset);
    assert_eq!(status, 200);
    assert!(headers.contains("content-type: text/javascript"));
    assert!(!javascript.is_empty());
    assert_eq!(get(port, "HEAD", asset).2.len(), 0);
    assert_eq!(get(port, "GET", "/api/lfg/dev-stats").0, 200);
    assert_eq!(get(port, "GET", "/api/not-a-route").0, 404);
    assert_eq!(get(port, "GET", "/assets/missing.js").0, 404);
}

#[test]
fn explicitly_configured_ui_overrides_embedded_assets() {
    let temp = tempfile::tempdir().unwrap();
    let ui = temp.path().join("configured-ui");
    fs::create_dir(&ui).unwrap();
    fs::write(ui.join("index.html"), "custom ui").unwrap();
    let (_server, port) = start(temp.path(), Some(&ui));
    assert_eq!(get(port, "GET", "/").2, b"custom ui");
}
