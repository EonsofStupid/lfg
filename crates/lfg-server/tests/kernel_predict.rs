use std::{
    io::Write,
    process::{Command, Stdio},
};

fn run(input: &str, maximum: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lfg-server"))
        .args(["kernel-predict", "--max-examples", maximum])
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
fn streams_a_bounded_batch_without_echoing_input() {
    let result = run(
        "{\"id\":\"a\",\"text\":\"logout doesn't clear the session\"}\n{\"id\":\"b\",\"text\":\"unmatched private text\"}\n",
        "2",
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout).unwrap();
    assert!(!output.contains("unmatched private text"));
    let rows: Vec<serde_json::Value> = output
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["result"]["state"], "complete");
    assert_eq!(rows[1]["result"]["state"], "escalate");
    assert_eq!(rows[1]["id"], "b");
    assert!(
        !run(
            "{\"id\":\"a\",\"text\":\"one\"}\n{\"id\":\"a\",\"text\":\"two\"}\n",
            "2"
        )
        .status
        .success()
    );
    assert!(
        !run(
            "{\"id\":\"a\",\"text\":\"one\"}\n{\"id\":\"b\",\"text\":\"two\"}\n",
            "1"
        )
        .status
        .success()
    );
}
