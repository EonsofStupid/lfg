//! Runs `lfg-server prompt-hook` as Claude Code and Codex would: a JSON payload on
//! stdin, context JSON (or nothing) on stdout, and always exit 0.

use std::io::Write;
use std::process::{Command, Stdio};

fn hook(client: &str, stdin: &str, journal: Option<&std::path::Path>) -> (bool, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_lfg-server"));
    command.args(["prompt-hook", "--client", client]);
    if let Some(path) = journal {
        command.arg("--journal").arg(path);
    }
    let mut child = command
        .env_remove("LFG_OPERATOR_MOOD_CONFIG")
        .env_remove("LFG_MOOD_JOURNAL")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    (
        output.status.success(),
        String::from_utf8(output.stdout).unwrap(),
    )
}

#[test]
fn frustrated_prompts_get_grounding_context_for_both_clients() {
    let claude = serde_json::json!({"session_id": "s-1", "transcript_path": "t", "cwd": "c",
        "hook_event_name": "UserPromptSubmit", "prompt": "why did you not run the tests?? I told you twice"});
    let (ok, out) = hook("claude", &claude.to_string(), None);
    assert!(ok);
    let value: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(
        value["hookSpecificOutput"]["hookEventName"],
        "UserPromptSubmit"
    );
    let context = value["hookSpecificOutput"]["additionalContext"]
        .as_str()
        .unwrap();
    assert!(context.contains("verified") && context.contains("next step"));
    let codex = serde_json::json!({"session_id": "s-2", "turn_id": "t-9", "model": "m",
        "hook_event_name": "UserPromptSubmit", "prompt": "WTF this is STILL BROKEN"});
    let (ok, out) = hook("codex", &codex.to_string(), None);
    assert!(ok && out.contains("additionalContext"));
    // A UTF-8 byte-order mark, as some shells add when piping, is ignored.
    let (ok, out) = hook("codex", &format!("\u{feff}{codex}"), None);
    assert!(ok && out.contains("additionalContext"));
}

#[test]
fn calm_malformed_or_unknown_input_adds_nothing_and_never_fails() {
    let calm =
        serde_json::json!({"session_id": "s", "prompt": "Add a retry to the webhook sender."});
    for (client, stdin) in [
        ("claude", calm.to_string()),
        ("claude", "not json".to_string()),
        ("claude", "{}".to_string()),
        ("other", calm.to_string()),
    ] {
        let (ok, out) = hook(client, &stdin, None);
        assert!(ok, "{client} {stdin}");
        assert!(out.trim().is_empty(), "{client} {stdin} -> {out}");
    }
}

#[test]
fn the_journal_keeps_readings_and_digests_but_never_the_words() {
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("mood.jsonl");
    let prompt = "zebra-canary why are you ignoring me!!";
    let payload = serde_json::json!({"session_id": "session-canary", "prompt": prompt});
    let (ok, _) = hook("claude", &payload.to_string(), Some(&journal));
    assert!(ok);
    let text = std::fs::read_to_string(&journal).unwrap();
    assert!(!text.contains("zebra-canary") && !text.contains("session-canary"));
    let entry: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
    assert_eq!(entry["state"], "frustrated");
    assert_eq!(entry["client"], "claude");
    assert_eq!(entry["promptChars"], prompt.chars().count());
    assert_eq!(entry["sessionSha256"].as_str().unwrap().len(), 64);
    assert!(
        entry["signals"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "why-you")
    );
}

#[test]
fn gemini_gets_before_agent_json_and_kimi_gets_plain_text() {
    let angry = serde_json::json!({"session_id": "s", "prompt": "why did you ignore me?? fix it"});
    let (ok, out) = hook("gemini", &angry.to_string(), None);
    assert!(ok);
    let value: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(value["hookSpecificOutput"]["hookEventName"], "BeforeAgent");
    assert!(value["hookSpecificOutput"]["additionalContext"].is_string());
    let (ok, out) = hook("kimi", &angry.to_string(), None);
    assert!(ok);
    assert!(out.starts_with("LFG operator context:") && !out.trim_start().starts_with('{'));
    // An undocumented prompt field name is still read.
    let other = serde_json::json!({"session_id": "s", "user_prompt": "WTF STILL BROKEN"});
    let (ok, out) = hook("kimi", &other.to_string(), None);
    assert!(ok && out.starts_with("LFG operator context:"));
    let calm = serde_json::json!({"session_id": "s", "prompt": "Rename the variable."});
    assert!(hook("gemini", &calm.to_string(), None).1.trim().is_empty());
}
