//! The `UserPromptSubmit` hook for Claude Code and Codex. Both send the prompt
//! as `prompt` on stdin and accept `hookSpecificOutput.additionalContext` on
//! stdout. When the prompt reads as frustrated, the hook returns grounding
//! context for the agent; otherwise it prints nothing. It never blocks a
//! prompt: any failure prints nothing and exits 0.
//!
//! An optional local journal records each reading for later learning. It keeps
//! pattern ids, sizes and SHA-256 digests only, never the prompt's words.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use lfg_core::operator_mood::OperatorMood;
use serde_json::json;
use sha2::{Digest, Sha256};

const MAX_STDIN_BYTES: u64 = 4 * 1024 * 1024;
const MAX_JOURNAL_BYTES: u64 = 64 * 1024 * 1024;

pub fn run(client: &str, journal: Option<PathBuf>) {
    // A hook must never stand between the operator and their agent.
    let _ = try_run(client, journal);
}

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn load_mood() -> Result<OperatorMood, Box<dyn std::error::Error>> {
    match std::env::var_os("LFG_OPERATOR_MOOD_CONFIG") {
        Some(path) => Ok(OperatorMood::from_json_str(&std::fs::read_to_string(
            path,
        )?)?),
        None => Ok(OperatorMood::bundled()),
    }
}

fn try_run(client: &str, journal: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    if !matches!(client, "claude" | "codex") {
        return Ok(());
    }
    let mut raw = Vec::new();
    std::io::stdin()
        .take(MAX_STDIN_BYTES + 1)
        .read_to_end(&mut raw)?;
    if raw.len() as u64 > MAX_STDIN_BYTES {
        return Ok(());
    }
    // Some shells prepend a UTF-8 byte-order mark when piping text.
    let body = raw.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&raw);
    let payload: serde_json::Value = serde_json::from_slice(body)?;
    let Some(prompt) = payload.get("prompt").and_then(serde_json::Value::as_str) else {
        return Ok(());
    };
    let mood = load_mood()?;
    let reading = mood.read(prompt);
    let journal = journal.or_else(|| std::env::var_os("LFG_MOOD_JOURNAL").map(PathBuf::from));
    if let Some(path) = journal {
        let field = |name: &str| {
            payload
                .get(name)
                .and_then(serde_json::Value::as_str)
                .map(sha)
        };
        let entry = json!({
            "schemaVersion": "lfg.operator-mood-reading.v1",
            "atUnixMs": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            "client": client,
            "sessionSha256": field("session_id"),
            "turnSha256": field("turn_id").or_else(|| field("prompt_id")),
            "promptSha256": sha(prompt),
            "promptChars": prompt.chars().count(),
            "state": reading.state,
            "signals": reading.signals,
            "moodRevision": mood.revision(),
            "moodSha256": mood.digest(),
        });
        let _ = append(&path, &entry);
    }
    if let Some(context) = mood.guidance(&reading) {
        let output = json!({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }});
        let mut stdout = std::io::stdout().lock();
        serde_json::to_writer(&mut stdout, &output)?;
        writeln!(stdout)?;
    }
    Ok(())
}

fn append(path: &Path, entry: &serde_json::Value) -> std::io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_JOURNAL_BYTES) {
        return Ok(());
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let mut line = serde_json::to_vec(entry)?;
    line.push(b'\n');
    file.write_all(&line)
}
