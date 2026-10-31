use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

fn predict(path: &Path, capture: bool) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
    command
        .args(["kernel-predict", "--max-examples", "2", "--journal"])
        .arg(path)
        .args(["--journal-max-bytes", "100000"]);
    if capture {
        command.arg("--capture-text");
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"id\":\"one\",\"text\":\"unmatched private fixture\"}\n")
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn active_writer_excludes_other_processes_until_stream_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal.jsonl");
    let mut child = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-predict", "--max-examples", "2", "--journal"])
        .arg(&path)
        .args(["--journal-max-bytes", "100000"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input
        .write_all(b"{\"id\":\"one\",\"text\":\"fixture\"}\n")
        .unwrap();
    input.flush().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&line).unwrap()["journal_sequence"],
        1
    );
    let status = || {
        Command::new(env!("CARGO_BIN_EXE_lfg-server"))
            .args(["kernel-journal-status", "--journal"])
            .arg(&path)
            .args(["--journal-max-bytes", "100000"])
            .output()
            .unwrap()
    };
    assert!(!status().status.success());
    drop(input);
    assert!(child.wait().unwrap().success());
    assert!(status().status.success());
}

#[test]
fn real_cli_journal_requires_attestation_and_reports_verified_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal.jsonl");
    let run = predict(&path, false);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let prediction: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert_eq!(prediction["journal_sequence"], 1);
    assert!(
        !std::fs::read_to_string(&path)
            .unwrap()
            .contains("unmatched private fixture")
    );
    let correction = |attest| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
        command
            .args(["kernel-correct", "--journal"])
            .arg(&path)
            .args([
                "--journal-max-bytes",
                "100000",
                "--observation",
                "1",
                "--node",
                "operation",
                "--choice",
                "chat",
                "--reviewer",
                "fixture-reviewer",
            ]);
        if attest {
            command.arg("--attest");
        }
        command.output().unwrap()
    };
    assert!(!correction(false).status.success());
    assert!(correction(true).status.success());
    let status = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-journal-status", "--journal"])
        .arg(&path)
        .args(["--journal-max-bytes", "100000"])
        .output()
        .unwrap();
    assert!(status.status.success());
    let summary: serde_json::Value = serde_json::from_slice(&status.stdout).unwrap();
    assert_eq!(summary["observations"], 1);
    assert_eq!(summary["corrections"], 1);
    assert_eq!(summary["capturedText"], 0);
}

#[test]
fn captured_correction_exports_only_as_an_unassigned_pending_candidate() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("journal.jsonl");
    let output = dir.path().join("candidate.jsonl");
    assert!(predict(&path, true).status.success());
    let show = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-journal-show", "--journal"])
        .arg(&path)
        .args(["--journal-max-bytes", "100000", "--observation", "1"])
        .output()
        .unwrap();
    let shown: serde_json::Value = serde_json::from_slice(&show.stdout).unwrap();
    assert!(shown["observation"]["capturedText"].is_null());
    assert_eq!(shown["capturedTextAvailable"], true);
    let correction = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-correct", "--journal"])
        .arg(&path)
        .args([
            "--journal-max-bytes",
            "100000",
            "--observation",
            "1",
            "--node",
            "operation",
            "--choice",
            "chat",
            "--reviewer",
            "fixture-reviewer",
            "--attest",
        ])
        .output()
        .unwrap();
    assert!(correction.status.success());
    let export = || {
        Command::new(env!("CARGO_BIN_EXE_lfg-server"))
            .args(["kernel-journal-export", "--journal"])
            .arg(&path)
            .args(["--journal-max-bytes", "100000", "--output"])
            .arg(&output)
            .args(["--output-max-bytes", "100000"])
            .output()
            .unwrap()
    };
    assert!(export().status.success());
    let candidate: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(candidate["status"], "pending_review");
    assert_eq!(candidate["proposedChoice"], "chat");
    assert!(candidate["project"].is_null() && candidate["split"].is_null());
    assert!(!export().status.success());
}
