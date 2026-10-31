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
fn review_queue_uses_verified_source_without_fabricating_labels() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("journal.jsonl");
    let source = dir.path().join("source.jsonl");
    let queue = dir.path().join("queue.jsonl");
    let mut child = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-predict", "--max-examples", "3", "--journal"])
        .arg(&journal)
        .args(["--journal-max-bytes", "100000"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let records =
        b"{\"id\":\"a\",\"text\":\"unmatched fixture\"}\n{\"id\":\"b\",\"text\":\"run command\"}\n{\"id\":\"c\",\"text\":\"logout run\"}\n";
    child.stdin.take().unwrap().write_all(records).unwrap();
    assert!(child.wait().unwrap().success());
    std::fs::write(&source, records).unwrap();
    let before = std::fs::read(&journal).unwrap();
    let invoke = |path: &Path, limit: &str, include: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
        command
            .args(["kernel-journal-review-queue", "--journal"])
            .arg(&journal)
            .args(["--journal-max-bytes", "100000", "--source-jsonl"])
            .arg(&source)
            .args(["--source-max-bytes", "100000", "--output"])
            .arg(path)
            .args(["--output-max-bytes", limit]);
        if include {
            command.arg("--include-text");
        }
        command.output().unwrap()
    };
    assert!(!invoke(&queue, "100000", false).status.success());
    assert!(!invoke(&queue, "1", true).status.success());
    assert!(!queue.exists());
    std::fs::write(&source, b"{\"id\":\"a\",\"text\":\"changed fixture\"}\n").unwrap();
    assert!(!invoke(&queue, "100000", true).status.success());
    assert!(!queue.exists());
    std::fs::write(&source, records).unwrap();
    let result = invoke(&queue, "100000", true);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!String::from_utf8_lossy(&result.stdout).contains("unmatched fixture"));
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["items"], 3);
    assert_eq!(report["baselineSelected"], 1);
    assert_eq!(report["baselineUnselected"], 2);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&queue)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["status"], "pending_review");
    assert_eq!(rows[0]["options"], serde_json::json!(["chat", "tool_call"]));
    assert!(rows[0]["baselineChoice"].is_null());
    assert_eq!(rows[1]["baselineChoice"], "tool_call");
    assert_eq!(rows[2]["baselineState"], "escalate");
    assert!(rows[2]["baselineChoice"].is_null());
    assert!(rows.iter().all(|row| row["reviewedChoice"].is_null()
        && row["reviewer"].is_null()
        && row["project"].is_null()
        && row["family"].is_null()
        && row["split"].is_null()));
    assert_eq!(std::fs::read(&journal).unwrap(), before);
    assert!(!invoke(&queue, "100000", true).status.success());

    let correction = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-correct", "--journal"])
        .arg(&journal)
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
            "test-reviewer",
            "--attest",
        ])
        .output()
        .unwrap();
    assert!(correction.status.success());
    let second = dir.path().join("queue2.jsonl");
    let result = invoke(&second, "100000", true);
    assert!(result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["items"], 2);
    assert_eq!(report["alreadyCorrected"], 1);
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

#[test]
fn digest_only_observation_can_use_a_verified_source_for_pending_export() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("journal.jsonl");
    let source = dir.path().join("source.jsonl");
    let output = dir.path().join("candidate.jsonl");
    assert!(predict(&journal, false).status.success());
    std::fs::write(
        &source,
        b"{\"id\":\"one\",\"text\":\"unmatched private fixture\",\"verified\":false}\n",
    )
    .unwrap();
    let reconcile = || {
        Command::new(env!("CARGO_BIN_EXE_lfg-server"))
            .args(["kernel-journal-reconcile", "--journal"])
            .arg(&journal)
            .args(["--journal-max-bytes", "100000", "--source-jsonl"])
            .arg(&source)
            .args(["--source-max-bytes", "100000"])
            .output()
            .unwrap()
    };
    let result = reconcile();
    assert!(result.status.success());
    assert!(!String::from_utf8_lossy(&result.stdout).contains("unmatched private fixture"));
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&result.stdout).unwrap()["matched"],
        1
    );
    let show = |include_text: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
        command
            .args(["kernel-journal-show", "--journal"])
            .arg(&journal)
            .args([
                "--journal-max-bytes",
                "100000",
                "--observation",
                "1",
                "--source-jsonl",
            ])
            .arg(&source)
            .args(["--source-max-bytes", "100000"]);
        if include_text {
            command.arg("--include-text");
        }
        command.output().unwrap()
    };
    let hidden: serde_json::Value = serde_json::from_slice(&show(false).stdout).unwrap();
    assert_eq!(hidden["sourceTextAvailable"], true);
    assert!(hidden["sourceText"].is_null());
    let shown: serde_json::Value = serde_json::from_slice(&show(true).stdout).unwrap();
    assert_eq!(shown["sourceText"], "unmatched private fixture");
    let correction = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-correct", "--journal"])
        .arg(&journal)
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
    let export = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-journal-export", "--journal"])
        .arg(&journal)
        .args(["--journal-max-bytes", "100000", "--source-jsonl"])
        .arg(&source)
        .args(["--source-max-bytes", "100000", "--output"])
        .arg(&output)
        .args(["--output-max-bytes", "100000"])
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let row: serde_json::Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(row["status"], "pending_review");
    assert_eq!(row["textOrigin"], "source_jsonl");
    assert!(row["split"].is_null());
    std::fs::write(&source, b"{\"id\":\"one\",\"text\":\"changed\"}\n").unwrap();
    assert!(!reconcile().status.success());
    assert!(!show(true).status.success());
}
