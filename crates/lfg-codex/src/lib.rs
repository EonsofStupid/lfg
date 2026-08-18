use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const ADAPTER_ID: &str = "openai.codex-cli";
pub const ADAPTER_VERSION: &str = "1.0.0";

#[derive(Clone, Debug)]
pub struct CodexRunner {
    binary: String,
    workspace: PathBuf,
    model: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexStatus {
    pub target_id: &'static str,
    pub adapter_id: &'static str,
    pub adapter_version: &'static str,
    pub provider: &'static str,
    pub display_name: &'static str,
    pub status: String,
    pub auth_class: &'static str,
    pub auth_method: String,
    pub cli_version: Option<String>,
    pub workspace: String,
    pub detail: String,
    pub capabilities: Vec<&'static str>,
}

impl CodexStatus {
    pub fn active(&self) -> bool {
        self.status == "active"
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderEvent {
    pub event_type: String,
    pub operation: String,
    pub status: String,
    pub native_event_id: Option<String>,
    pub native_payload_length: usize,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionResult {
    pub final_response: String,
    pub usage: Usage,
    pub events: Vec<ProviderEvent>,
}

#[derive(Debug, Error)]
pub enum CodexError {
    #[error("Codex subscription provider is not active")]
    Inactive,
    #[error("Codex execution failed: {0}")]
    Execution(&'static str),
    #[error("Codex emitted an invalid JSON event stream")]
    InvalidStream,
}

impl CodexRunner {
    pub fn new(
        binary: impl Into<String>,
        workspace: impl Into<PathBuf>,
        model: Option<String>,
    ) -> Self {
        let binary = binary.into();
        Self {
            binary: if binary.trim().is_empty() {
                "codex".into()
            } else {
                binary
            },
            workspace: workspace.into(),
            model: model.filter(|value| !value.trim().is_empty()),
        }
    }

    pub fn from_environment(workspace: impl Into<PathBuf>) -> Self {
        Self::new(
            env::var("LFG_CODEX_BINARY").unwrap_or_else(|_| "codex".into()),
            workspace,
            env::var("LFG_CODEX_MODEL").ok(),
        )
    }

    pub fn probe(&self) -> CodexStatus {
        let mut status = CodexStatus {
            target_id: "openai-codex-seat",
            adapter_id: ADAPTER_ID,
            adapter_version: ADAPTER_VERSION,
            provider: "openai",
            display_name: "OpenAI Codex subscription",
            status: "missing".into(),
            auth_class: "subscription-seat",
            auth_method: "unknown".into(),
            cli_version: None,
            workspace: self
                .workspace
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("workspace")
                .into(),
            detail: "Codex CLI is not installed".into(),
            capabilities: vec![
                "agent.invoke",
                "filesystem.read",
                "command.read-only",
                "stream.jsonl",
            ],
        };
        let Some(binary) = find_binary(&self.binary) else {
            return status;
        };
        let version = self.command(&binary).arg("--version").output();
        status.cli_version = version
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|value| !value.is_empty());
        let login = self.command(&binary).args(["login", "status"]).output();
        let Ok(output) = login else {
            status.status = "unavailable".into();
            status.detail = "Codex CLI could not be probed".into();
            return status;
        };
        if !output.status.success() {
            status.status = "unauthenticated".into();
            status.detail = "Codex CLI is installed but has no usable login".into();
            return status;
        }
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .to_lowercase();
        if !combined.contains("chatgpt") {
            status.status = "unsupported-auth".into();
            status.auth_method = "non-subscription".into();
            status.detail =
                "Codex is authenticated, but not through a ChatGPT subscription seat".into();
            return status;
        }
        status.status = "active".into();
        status.auth_method = "chatgpt".into();
        status.detail = "Subscription seat is ready for an explicit read-only LFG run".into();
        status
    }

    pub fn execute(&self, prompt: &str) -> Result<ExecutionResult, CodexError> {
        if !self.probe().active() {
            return Err(CodexError::Inactive);
        }
        let binary = find_binary(&self.binary).ok_or(CodexError::Inactive)?;
        let mut command = self.command(&binary);
        command
            .args([
                "exec",
                "--json",
                "--ephemeral",
                "--ignore-user-config",
                "--sandbox",
                "read-only",
                "--cd",
            ])
            .arg(&self.workspace)
            .args(["--config", "approval_policy=\"never\"", "--color", "never"]);
        if let Some(model) = &self.model {
            command.args(["--model", model]);
        }
        command
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command
            .spawn()
            .map_err(|_| CodexError::Execution("could not start process"))?;
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .ok_or(CodexError::Execution("stdin unavailable"))?
            .write_all(prompt.as_bytes())
            .map_err(|_| CodexError::Execution("could not send request"))?;
        drop(child.stdin.take());
        let stdout = child
            .stdout
            .take()
            .ok_or(CodexError::Execution("stdout unavailable"))?;
        let result = consume_jsonl(stdout)?;
        let status = child
            .wait()
            .map_err(|_| CodexError::Execution("could not wait for process"))?;
        if !status.success() {
            return Err(CodexError::Execution("process exited unsuccessfully"));
        }
        if result.final_response.trim().is_empty() {
            return Err(CodexError::Execution("final response missing"));
        }
        Ok(result)
    }

    fn command(&self, binary: &Path) -> Command {
        let mut command = Command::new(binary);
        command
            .current_dir(&self.workspace)
            .env_clear()
            .envs(safe_environment());
        command
    }
}

fn find_binary(binary: &str) -> Option<PathBuf> {
    if binary.contains(std::path::MAIN_SEPARATOR) {
        return Path::new(binary).is_file().then(|| PathBuf::from(binary));
    }
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|path| path.join(binary))
            .find(|path| path.is_file())
    })
}

fn safe_environment() -> BTreeMap<String, String> {
    const ALLOWED: &[&str] = &[
        "PATH",
        "HOME",
        "CODEX_HOME",
        "LANG",
        "LC_ALL",
        "TERM",
        "TMPDIR",
        "USER",
        "LOGNAME",
        "SHELL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CODEX_CA_CERTIFICATE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ];
    env::vars()
        .filter(|(name, _)| ALLOWED.contains(&name.as_str()) || name.starts_with("LC_"))
        .collect()
}

#[derive(Deserialize)]
struct RawEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    item: Option<RawItem>,
    #[serde(default)]
    usage: RawUsage,
}
#[derive(Deserialize)]
struct RawItem {
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    text: String,
}
#[derive(Default, Deserialize)]
struct RawUsage {
    #[serde(default)]
    input_tokens: i64,
    #[serde(default)]
    cached_input_tokens: i64,
    #[serde(default)]
    output_tokens: i64,
    #[serde(default)]
    reasoning_output_tokens: i64,
}

pub fn consume_jsonl(reader: impl Read) -> Result<ExecutionResult, CodexError> {
    let mut result = ExecutionResult::default();
    let mut terminal = false;
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|_| CodexError::InvalidStream)?;
        let raw: RawEvent = serde_json::from_str(&line).map_err(|_| CodexError::InvalidStream)?;
        let mut event = ProviderEvent {
            event_type: raw.event_type.clone(),
            operation: operation_for_event(&raw.event_type).into(),
            status: status_for_event(&raw.event_type).into(),
            native_event_id: None,
            native_payload_length: line.len(),
            attributes: BTreeMap::new(),
        };
        if let Some(item) = raw.item {
            event.operation = operation_for_item(&item.item_type).into();
            event.native_event_id = (!item.id.is_empty()).then_some(item.id);
            if !item.status.is_empty() {
                event.status = item.status;
            }
            event
                .attributes
                .insert("item_type".into(), item.item_type.clone());
            if raw.event_type == "item.completed" && item.item_type == "agent_message" {
                result.final_response = item.text;
            }
        }
        if raw.event_type == "turn.completed" {
            terminal = true;
            result.usage = Usage {
                input_tokens: raw.usage.input_tokens,
                cached_input_tokens: raw.usage.cached_input_tokens,
                output_tokens: raw.usage.output_tokens,
                reasoning_output_tokens: raw.usage.reasoning_output_tokens,
            };
        } else if raw.event_type == "turn.failed" || raw.event_type == "error" {
            terminal = false;
        }
        result.events.push(event);
    }
    if !terminal {
        return Err(CodexError::Execution("successful terminal event missing"));
    }
    Ok(result)
}

fn operation_for_event(value: &str) -> &'static str {
    match value {
        "thread.started" => "provider.thread",
        "turn.started" | "turn.completed" | "turn.failed" => "model.turn",
        "error" => "provider.error",
        _ => "provider.event",
    }
}
fn operation_for_item(value: &str) -> &'static str {
    match value {
        "command_execution" | "mcp_tool_call" | "collab_tool_call" => "tool.execute",
        "file_change" => "artifact.mutate",
        "web_search" => "web.search",
        "reasoning" => "agent.reasoning",
        "agent_message" => "agent.message",
        "todo_list" => "agent.plan",
        _ => "agent.item",
    }
}
fn status_for_event(value: &str) -> &'static str {
    if value.ends_with(".started") {
        "started"
    } else if value.ends_with(".completed") {
        "completed"
    } else if value.ends_with(".failed") || value == "error" {
        "error"
    } else {
        "observed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_success_stream_without_exposing_reasoning_text_in_events() {
        let stream = concat!(
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"secret-free-id\",\"type\":\"reasoning\",\"text\":\"private chain\"}}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"answer\",\"type\":\"agent_message\",\"text\":\"final answer\"}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}\n"
        );
        let result = consume_jsonl(stream.as_bytes()).unwrap();
        assert_eq!(result.final_response, "final answer");
        assert_eq!(result.usage.output_tokens, 2);
        let serialized = serde_json::to_string(&result.events).unwrap();
        assert!(!serialized.contains("private chain"));
        assert!(!serialized.contains("final answer"));
    }

    #[test]
    fn rejects_failed_or_malformed_streams() {
        assert!(consume_jsonl(b"{bad json}\n".as_slice()).is_err());
        assert!(consume_jsonl(b"{\"type\":\"turn.failed\"}\n".as_slice()).is_err());
    }

    #[test]
    fn environment_allowlist_excludes_api_keys() {
        let environment = safe_environment();
        assert!(!environment.contains_key("OPENAI_API_KEY"));
        assert!(!environment.keys().any(|name| name.contains("SECRET")));
    }
}
