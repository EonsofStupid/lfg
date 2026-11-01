//! Learns which models exist and how they are used from the provider tools' own
//! local files, and keeps that current as the user works.
//!
//! Only model names, dates and counts are kept. Prompt text is scored in memory
//! and dropped, nothing is written to disk, and nothing leaves the machine.
//! Files are read incrementally, so a refresh costs only what was appended.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lfg_core::PreflightRuleConfig;
use lfg_core::assessment::{
    AssessmentConfig, BracketUsage, Knowledge, KnowledgeSource, KnownModel, ObservedSubscription,
    bracket_for,
};
use serde::Deserialize;
use serde_json::Value;

const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;
const READ_BUFFER_BYTES: usize = 1024 * 1024;
const MAIN_QUERY_SOURCES: [&str; 2] = ["repl_main_thread", "sdk"];
/// Text the tools themselves send that is still marked as coming from a person.
const NOT_TYPED_PREFIXES: [&str; 5] = [
    "<command",
    "<local-",
    "<system-reminder",
    "<task-notification",
    "This session is being continued",
];
const CODEX: &str = "codex";
const CLAUDE: &str = "claude";

/// Where the provider tools keep their files on this machine.
#[derive(Clone, Debug, Default)]
pub struct Locations {
    pub codex_home: Option<PathBuf>,
    pub claude_home: Option<PathBuf>,
    pub claude_config: Option<PathBuf>,
}

impl Locations {
    pub fn from_environment() -> Self {
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from);
        let claude_dir = std::env::var_os("CLAUDE_CONFIG_DIR").map(PathBuf::from);
        Self {
            codex_home: std::env::var_os("CODEX_HOME")
                .map(PathBuf::from)
                .or_else(|| home.as_ref().map(|path| path.join(".codex"))),
            claude_home: claude_dir
                .clone()
                .or_else(|| home.as_ref().map(|path| path.join(".claude"))),
            claude_config: claude_dir
                .map(|path| path.join(".claude.json"))
                .or_else(|| home.as_ref().map(|path| path.join(".claude.json"))),
        }
    }
}

type UsageKey = (String, usize, Option<String>, String);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Codex,
    Claude,
}

impl Kind {
    fn subscription(self) -> &'static str {
        match self {
            Self::Codex => CODEX,
            Self::Claude => CLAUDE,
        }
    }
}

#[derive(Debug)]
struct FileState {
    kind: Kind,
    offset: u64,
    /// Automated sessions, such as Codex guardian reviews, are not the user's choice.
    skip: bool,
    model: Option<String>,
    effort: Option<String>,
    pending: Vec<(usize, String)>,
    counts: BTreeMap<UsageKey, u64>,
}

impl FileState {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            offset: 0,
            skip: false,
            model: None,
            effort: None,
            pending: Vec::new(),
            counts: BTreeMap::new(),
        }
    }

    fn count(&mut self, model: &str, bracket: usize, effort: Option<String>, day: String) {
        *self
            .counts
            .entry((model.to_string(), bracket, effort, day))
            .or_insert(0) += 1;
    }
}

pub struct Learner {
    config: Arc<AssessmentConfig>,
    rules: Arc<PreflightRuleConfig>,
    locations: Locations,
    files: HashMap<PathBuf, FileState>,
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    memchr::memmem::find(haystack, needle).is_some()
}

fn day_of(row: &Value) -> Option<String> {
    row["timestamp"]
        .as_str()
        .and_then(|stamp| stamp.get(..10))
        .map(str::to_string)
}

fn truthy(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
}

fn clean_model(model: &str) -> Option<&str> {
    let name = model.split('[').next().unwrap_or(model).trim();
    (!name.is_empty() && !name.starts_with('<')).then_some(name)
}

/// The text of a genuine user prompt in a Codex session, matching the rule in
/// `ml/lfg-encoder/inventory_codex_sessions.py`.
fn codex_prompt(payload: &Value) -> Option<&str> {
    if payload["type"] != "message" || payload["role"] != "user" {
        return None;
    }
    let kinds =
        payload["internal_chat_message_metadata_passthrough"]["content_item_kinds"].as_array()?;
    let content = payload["content"].as_array()?;
    if kinds.len() != 1 || kinds[0] != "user.text" || content.len() != 1 {
        return None;
    }
    if content[0]["type"] != "input_text" {
        return None;
    }
    content[0]["text"]
        .as_str()
        .filter(|text| !text.trim().is_empty())
}

/// The text of a genuine user prompt in a Claude session, matching the rule in
/// `ml/lfg-encoder/inventory_claude_sessions.py`.
fn claude_prompt(row: &Value) -> Option<&str> {
    if row["type"] != "user"
        || row["turnOrigin"] != "human"
        || row["isSidechain"] != Value::Bool(false)
        || truthy(&row["isMeta"])
        || truthy(&row["isCompactSummary"])
        || truthy(&row["isVisibleInTranscriptOnly"])
        || row.get("sourceToolAssistantUUID").is_some()
        || row.get("toolUseResult").is_some()
        || row["promptSource"] == "system"
        || row["message"]["role"] != "user"
    {
        return None;
    }
    let content = &row["message"]["content"];
    let text = match content {
        Value::String(text) => text.as_str(),
        Value::Array(blocks) if blocks.len() == 1 && blocks[0]["type"] == "text" => {
            blocks[0]["text"].as_str()?
        }
        _ => return None,
    };
    let trimmed = text.trim_start();
    (!trimmed.is_empty()
        && !NOT_TYPED_PREFIXES
            .iter()
            .any(|prefix| trimmed.starts_with(prefix)))
    .then_some(text)
}

fn read_new_lines(
    path: &Path,
    state: &mut FileState,
    mut handle: impl FnMut(&[u8], &mut FileState),
) -> std::io::Result<()> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(state.offset))?;
    let mut reader = BufReader::with_capacity(READ_BUFFER_BYTES, file);
    let mut line = Vec::new();
    let mut oversized = false;
    let mut line_bytes = 0u64;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            // A partial last line stays unread until the writer finishes it.
            return Ok(());
        }
        let newline = memchr::memchr(b'\n', available);
        let take = newline.map_or(available.len(), |index| index + 1);
        if !oversized {
            if line.len() + take > MAX_LINE_BYTES {
                oversized = true;
                line = Vec::new();
            } else {
                line.extend_from_slice(&available[..take]);
            }
        }
        reader.consume(take);
        line_bytes += take as u64;
        if newline.is_some() {
            if !oversized {
                handle(&line, state);
            }
            state.offset += line_bytes;
            line_bytes = 0;
            line.clear();
            oversized = false;
            if state.skip {
                return Ok(());
            }
        }
    }
}

fn jsonl_files_under(root: &Path, recursive: bool, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() && recursive {
            jsonl_files_under(&path, true, found);
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
            found.push(path);
        }
    }
}

fn session_files(locations: &Locations) -> Vec<(Kind, PathBuf)> {
    let mut sessions = Vec::new();
    if let Some(home) = &locations.codex_home {
        let mut found = Vec::new();
        jsonl_files_under(&home.join("sessions"), true, &mut found);
        sessions.extend(found.into_iter().map(|path| (Kind::Codex, path)));
    }
    if let Some(home) = &locations.claude_home
        && let Ok(projects) = std::fs::read_dir(home.join("projects"))
    {
        for project in projects.flatten() {
            if project.file_type().is_ok_and(|kind| kind.is_dir()) {
                let mut found = Vec::new();
                jsonl_files_under(&project.path(), false, &mut found);
                sessions.extend(found.into_iter().map(|path| (Kind::Claude, path)));
            }
        }
    }
    sessions
}

/// Whole days since the civil epoch, for comparing `YYYY-MM-DD` dates.
fn day_number(day: &str) -> Option<i64> {
    let mut parts = day.get(..10)?.split('-');
    let year: i64 = parts.next()?.parse().ok()?;
    let month: i64 = parts.next()?.parse().ok()?;
    let date: i64 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&date) {
        return None;
    }
    let shifted = if month <= 2 { year - 1 } else { year };
    let era = shifted.div_euclid(400);
    let year_of_era = shifted.rem_euclid(400);
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + date - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Some(era * 146_097 + day_of_era)
}

#[derive(Default)]
struct ProviderList {
    fetched_at: Option<String>,
    models: Vec<KnownModel>,
}

/// Codex's own list of what this account is offered, refreshed by the Codex CLI.
fn read_codex_list(home: &Path) -> Option<ProviderList> {
    let text = std::fs::read_to_string(home.join("models_cache.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let models = value["models"]
        .as_array()?
        .iter()
        .filter(|model| model["visibility"] == "list")
        .filter_map(|model| {
            let slug = model["slug"].as_str()?;
            Some(KnownModel {
                subscription_id: CODEX.into(),
                model: slug.to_string(),
                label: model["display_name"].as_str().unwrap_or(slug).to_string(),
                description: model["description"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                priority: model["priority"]
                    .as_u64()
                    .and_then(|p| u32::try_from(p).ok()),
                efforts: model["supported_reasoning_levels"]
                    .as_array()
                    .map(|levels| {
                        levels
                            .iter()
                            .filter_map(|level| level["effort"].as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
                origin: "provider-list".into(),
                available: true,
                ..KnownModel::default()
            })
        })
        .collect();
    Some(ProviderList {
        fetched_at: value["fetched_at"]
            .as_str()
            .and_then(|stamp| stamp.get(..10))
            .map(str::to_string),
        models,
    })
}

#[derive(Default)]
struct ClaudeAccount {
    plan: Option<String>,
    models: Vec<KnownModel>,
}

fn title_words(value: &str) -> String {
    value
        .split('_')
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut letters = word.chars();
            letters.next().map_or_else(String::new, |first| {
                first.to_uppercase().chain(letters).collect::<String>()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeConfigProjection {
    oauth_account: Option<ClaudePlanProjection>,
    additional_model_options_cache: Option<Vec<ClaudeModelProjection>>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudePlanProjection {
    organization_type: Option<String>,
    organization_rate_limit_tier: Option<String>,
}

#[derive(Deserialize)]
struct ClaudeModelProjection {
    value: Option<String>,
    label: Option<String>,
    description: Option<String>,
}

/// Stream the config while retaining only the plan and model-option fields.
/// The parser still reads the file, but does not retain other account fields.
fn read_claude_account(path: &Path) -> Option<ClaudeAccount> {
    let reader = BufReader::new(File::open(path).ok()?);
    let value: ClaudeConfigProjection = serde_json::from_reader(reader).ok()?;
    let account = value.oauth_account.unwrap_or_default();
    let plan = account.organization_type.as_deref().map(|kind| {
        let name = title_words(kind);
        let multiplier = account
            .organization_rate_limit_tier
            .as_deref()
            .and_then(|tier| tier.rsplit('_').next())
            .filter(|word| word.ends_with('x') && word[..word.len() - 1].parse::<u32>().is_ok());
        multiplier.map_or(name.clone(), |value| format!("{name} {value}"))
    });
    let models = value
        .additional_model_options_cache
        .as_deref()
        .map(|options| {
            options
                .iter()
                .filter_map(|option| {
                    let model = clean_model(option.value.as_deref()?)?;
                    Some(KnownModel {
                        subscription_id: CLAUDE.into(),
                        model: model.to_string(),
                        label: option.label.as_deref().unwrap_or(model).to_string(),
                        description: option
                            .description
                            .as_deref()
                            .unwrap_or_default()
                            .to_string(),
                        origin: "provider-list".into(),
                        available: true,
                        ..KnownModel::default()
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(ClaudeAccount { plan, models })
}

#[derive(Default)]
struct Tally {
    prompts: u64,
    weighted: f64,
    efforts: BTreeMap<String, u64>,
}

impl Learner {
    pub fn new(
        config: Arc<AssessmentConfig>,
        rules: Arc<PreflightRuleConfig>,
        locations: Locations,
    ) -> Self {
        Self {
            config,
            rules,
            locations,
            files: HashMap::new(),
        }
    }

    fn scan_codex_line(
        config: &AssessmentConfig,
        rules: &PreflightRuleConfig,
        line: &[u8],
        state: &mut FileState,
    ) {
        let head = &line[..line.len().min(512)];
        if contains(head, b"\"session_meta\"") {
            if let Ok(row) = serde_json::from_slice::<Value>(line) {
                state.skip = row["payload"]["source"].is_object();
            }
        } else if contains(head, b"\"turn_context\"") {
            if let Ok(row) = serde_json::from_slice::<Value>(line) {
                state.model = row["payload"]["model"]
                    .as_str()
                    .and_then(clean_model)
                    .map(str::to_string);
                state.effort = row["payload"]["effort"].as_str().map(str::to_string);
            }
        } else if contains(head, b"\"response_item\"")
            && contains(head, b"\"role\":\"user\"")
            && let Ok(row) = serde_json::from_slice::<Value>(line)
            && let Some(text) = codex_prompt(&row["payload"])
            && let Some(day) = day_of(&row)
            && let Some(model) = state.model.clone()
        {
            let bracket = bracket_for(config, rules, text);
            let effort = state.effort.clone();
            state.count(&model, bracket, effort, day);
        }
    }

    fn scan_claude_line(
        config: &AssessmentConfig,
        rules: &PreflightRuleConfig,
        line: &[u8],
        state: &mut FileState,
    ) {
        if contains(line, b"\"turnOrigin\":\"human\"") {
            if let Ok(row) = serde_json::from_slice::<Value>(line)
                && let Some(text) = claude_prompt(&row)
                && let Some(day) = day_of(&row)
            {
                state.pending.push((bracket_for(config, rules, text), day));
            }
            return;
        }
        if state.pending.is_empty() {
            return;
        }
        let attribution = if contains(line, b"\"type\":\"api-request\"") {
            serde_json::from_slice::<Value>(line).ok().and_then(|row| {
                let source = row["querySource"].as_str()?;
                if row["type"] != "api-request" || !MAIN_QUERY_SOURCES.contains(&source) {
                    return None;
                }
                let model = clean_model(row["params"]["model"].as_str()?)?.to_string();
                let effort = row["params"]["output_config"]["effort"]
                    .as_str()
                    .map(str::to_string);
                Some((model, effort))
            })
        } else if contains(line, b"\"type\":\"assistant\"") {
            // Older sessions have no request records, only the response's model.
            serde_json::from_slice::<Value>(line).ok().and_then(|row| {
                if row["type"] != "assistant" || row["isSidechain"] != Value::Bool(false) {
                    return None;
                }
                let model = clean_model(row["message"]["model"].as_str()?)?.to_string();
                Some((model, None))
            })
        } else {
            None
        };
        if let Some((model, effort)) = attribution {
            for (bracket, day) in std::mem::take(&mut state.pending) {
                state.count(&model, bracket, effort.clone(), day);
            }
        }
    }

    fn scan(&mut self, path: &Path, kind: Kind) {
        let Ok(metadata) = std::fs::metadata(path) else {
            return;
        };
        let len = metadata.len();
        let state = self
            .files
            .entry(path.to_path_buf())
            .or_insert_with(|| FileState::new(kind));
        if len < state.offset {
            *state = FileState::new(kind);
        }
        if state.skip || len == state.offset {
            return;
        }
        let config = &self.config;
        let rules = &self.rules;
        // A read error keeps what was already learned; the next refresh retries.
        let _ = read_new_lines(path, state, |line, state| match kind {
            Kind::Codex => Self::scan_codex_line(config, rules, line, state),
            Kind::Claude => Self::scan_claude_line(config, rules, line, state),
        });
    }

    /// Reads what changed and rebuilds what LFG knows. Blocking.
    pub fn refresh(&mut self) -> Knowledge {
        let now = lfg_core::now_rfc3339();
        let today = day_number(&now).unwrap_or(0);
        let sessions = session_files(&self.locations);
        let current: HashSet<&PathBuf> = sessions.iter().map(|(_, path)| path).collect();
        self.files.retain(|path, _| current.contains(path));
        for (kind, path) in &sessions {
            self.scan(path, *kind);
        }

        let half_life = self.config.matching.usage_half_life_days;
        let mut usage: BTreeMap<(&'static str, String), BTreeMap<usize, Tally>> = BTreeMap::new();
        let mut last_used: BTreeMap<(&'static str, String), String> = BTreeMap::new();
        for state in self.files.values() {
            let subscription = state.kind.subscription();
            for ((model, bracket, effort, day), count) in &state.counts {
                let age = day_number(day).map_or(0, |number| (today - number).max(0));
                let tally = usage
                    .entry((subscription, model.clone()))
                    .or_default()
                    .entry(*bracket)
                    .or_default();
                tally.prompts += count;
                tally.weighted += *count as f64 * 0.5f64.powf(age as f64 / half_life);
                if let Some(effort) = effort {
                    *tally.efforts.entry(effort.clone()).or_insert(0) += count;
                }
                let latest = last_used.entry((subscription, model.clone())).or_default();
                if day.as_str() > latest.as_str() {
                    latest.clone_from(day);
                }
            }
        }

        let codex_list = self
            .locations
            .codex_home
            .as_deref()
            .and_then(read_codex_list);
        let claude_account = self
            .locations
            .claude_config
            .as_deref()
            .and_then(read_claude_account);
        let codex_has_list = codex_list.is_some();
        let mut models: Vec<KnownModel> = codex_list
            .as_ref()
            .map(|list| list.models.clone())
            .unwrap_or_default();
        if let Some(account) = &claude_account {
            models.extend(account.models.iter().cloned());
        }
        for (subscription, model) in usage.keys() {
            if !models
                .iter()
                .any(|known| known.subscription_id == *subscription && known.model == *model)
            {
                models.push(KnownModel {
                    subscription_id: (*subscription).to_string(),
                    model: model.clone(),
                    label: model.clone(),
                    origin: "session".into(),
                    // A provider list that omits a model means it is no longer offered.
                    available: !(*subscription == CODEX && codex_has_list),
                    ..KnownModel::default()
                });
            }
        }
        for known in &mut models {
            let key = (
                if known.subscription_id == CODEX {
                    CODEX
                } else {
                    CLAUDE
                },
                known.model.clone(),
            );
            known.last_used = last_used.get(&key).cloned();
            if let Some(brackets) = usage.get(&key) {
                known.usage = brackets
                    .iter()
                    .filter_map(|(bracket, tally)| {
                        Some(BracketUsage {
                            bracket: self.config.brackets.get(*bracket)?.id.clone(),
                            prompts: tally.prompts,
                            weighted: tally.weighted,
                            efforts: tally.efforts.clone(),
                        })
                    })
                    .collect();
            }
        }
        models.sort_by(|a, b| {
            (&a.subscription_id, a.priority.unwrap_or(u32::MAX), &a.model).cmp(&(
                &b.subscription_id,
                b.priority.unwrap_or(u32::MAX),
                &b.model,
            ))
        });

        let prompts_for = |subscription: &str| -> u64 {
            usage
                .iter()
                .filter(|((id, _), _)| *id == subscription)
                .flat_map(|(_, brackets)| brackets.values())
                .map(|tally| tally.prompts)
                .sum()
        };
        let latest_for = |subscription: &str| {
            last_used
                .iter()
                .filter(|((id, _), _)| *id == subscription)
                .map(|(_, day)| day.clone())
                .max()
        };
        let mut subscriptions = Vec::new();
        for (id, plan, listed_at) in [
            (
                CLAUDE,
                claude_account.as_ref().and_then(|a| a.plan.clone()),
                None,
            ),
            (
                CODEX,
                None,
                codex_list.as_ref().and_then(|list| {
                    let fetched = list.fetched_at.clone()?;
                    let age = today - day_number(&fetched)?;
                    (age <= i64::from(self.config.matching.seen_within_days)).then_some(fetched)
                }),
            ),
        ] {
            let used = latest_for(id);
            if plan.is_some() || listed_at.is_some() || used.is_some() {
                subscriptions.push(ObservedSubscription {
                    id: id.into(),
                    plan,
                    listed_at,
                    days_since_used: used
                        .as_deref()
                        .and_then(day_number)
                        .and_then(|number| u32::try_from((today - number).max(0)).ok()),
                    last_used: used,
                });
            }
        }

        let source = |id: &str, kind: Kind, subscription: &str| {
            let files = sessions.iter().filter(|(k, _)| *k == kind).count();
            KnowledgeSource {
                id: id.into(),
                files,
                prompts: prompts_for(subscription),
                status: if files > 0 { "read" } else { "not-found" }.into(),
            }
        };
        let sources = vec![
            source("codex-sessions", Kind::Codex, CODEX),
            source("claude-sessions", Kind::Claude, CLAUDE),
        ];
        Knowledge {
            state: "ready".into(),
            refreshed_at: Some(now),
            prompts_seen: sources.iter().map(|s| s.prompts).sum(),
            sources,
            subscriptions,
            models,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    const SECRET_PROMPT: &str = "Fix the zebra-password-canary bug and add tests";

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("lfg-knowledge-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn append(path: &Path, lines: &[Value]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        for line in lines {
            writeln!(file, "{line}").unwrap();
        }
    }

    fn learner(root: &Path) -> Learner {
        Learner::new(
            Arc::new(AssessmentConfig::default_alpha()),
            Arc::new(
                PreflightRuleConfig::from_json_str(include_str!(
                    "../../lfg-core/config/preflight.v1.json"
                ))
                .unwrap(),
            ),
            Locations {
                codex_home: Some(root.join("codex")),
                claude_home: Some(root.join("claude")),
                claude_config: Some(root.join("claude.json")),
            },
        )
    }

    fn today() -> String {
        lfg_core::now_rfc3339()
    }

    fn codex_meta(source: Value) -> Value {
        serde_json::json!({"timestamp": today(), "type": "session_meta", "payload": {"source": source}})
    }

    fn codex_turn(model: &str, effort: &str) -> Value {
        serde_json::json!({"timestamp": today(), "type": "turn_context", "payload": {"model": model, "effort": effort}})
    }

    fn codex_user(text: &str, kind: &str) -> Value {
        serde_json::json!({"timestamp": today(), "type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": text}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": [kind]}
        }})
    }

    fn claude_user(text: &str, origin: &str) -> Value {
        serde_json::json!({"timestamp": today(), "type": "user", "turnOrigin": origin,
            "isSidechain": false, "message": {"role": "user", "content": text}})
    }

    fn claude_request(model: &str, effort: &str, source: &str) -> Value {
        serde_json::json!({"timestamp": today(), "type": "api-request", "querySource": source,
            "params": {"model": model, "output_config": {"effort": effort}}})
    }

    fn find<'a>(knowledge: &'a Knowledge, model: &str) -> &'a KnownModel {
        knowledge.models.iter().find(|m| m.model == model).unwrap()
    }

    #[test]
    fn counts_genuine_codex_prompts_and_reads_only_what_was_appended() {
        let root = scratch("codex");
        let session = root.join("codex/sessions/2026/a.jsonl");
        append(
            &session,
            &[
                codex_meta(serde_json::json!("vscode")),
                codex_turn("gpt-6-astra", "high"),
                codex_user(SECRET_PROMPT, "user.text"),
                codex_user("<environment_context>", "environments.environment_context"),
            ],
        );
        append(
            &root.join("codex/sessions/2026/review.jsonl"),
            &[
                codex_meta(serde_json::json!({"subagent": "guardian_review"})),
                codex_turn("codex-auto-review", "low"),
                codex_user("review this", "user.text"),
            ],
        );
        let mut learner = learner(&root);
        let first = learner.refresh();
        assert_eq!(first.state, "ready");
        assert_eq!(first.prompts_seen, 1);
        assert!(first.models.iter().all(|m| m.model != "codex-auto-review"));
        let astra = find(&first, "gpt-6-astra");
        assert_eq!(astra.usage.len(), 1);
        assert_eq!(astra.usage[0].prompts, 1);
        assert_eq!(astra.usage[0].efforts.get("high"), Some(&1));
        assert!(astra.available);

        append(
            &session,
            &[
                codex_turn("gpt-6.1-terra", "low"),
                codex_user("hey thanks", "user.text"),
            ],
        );
        let second = learner.refresh();
        assert_eq!(second.prompts_seen, 2);
        assert_eq!(find(&second, "gpt-6-astra").usage[0].prompts, 1);
        assert_eq!(find(&second, "gpt-6.1-terra").usage[0].bracket, "trivial");
        assert_eq!(second.subscriptions[0].id, "codex");
        assert_eq!(second.subscriptions[0].days_since_used, Some(0));
    }

    #[test]
    fn counts_genuine_claude_prompts_with_the_model_and_effort_of_the_next_request() {
        let root = scratch("claude");
        let session = root.join("claude/projects/project/a.jsonl");
        append(
            &session,
            &[
                claude_user("<command-name>/clear</command-name>", "human"),
                claude_user(
                    "This session is being continued from a previous one",
                    "human",
                ),
                claude_user("from a background task", "task_notification"),
                claude_user(SECRET_PROMPT, "human"),
                claude_request("claude-haiku-5-5", "low", "compact"),
                claude_request("claude-opus-5-5[1m]", "xhigh", "repl_main_thread"),
                serde_json::json!({"timestamp": today(), "type": "user", "turnOrigin": "human",
                    "isSidechain": true, "message": {"role": "user", "content": "from a subagent"}}),
            ],
        );
        append(
            &root.join("claude/projects/project/b.jsonl"),
            &[
                claude_user("what is rust", "human"),
                serde_json::json!({"timestamp": today(), "type": "assistant", "isSidechain": false,
                    "message": {"model": "claude-fable-5-1"}}),
                claude_user("next", "human"),
                serde_json::json!({"timestamp": today(), "type": "assistant", "isSidechain": false,
                    "message": {"model": "<synthetic>"}}),
            ],
        );
        let knowledge = learner(&root).refresh();
        assert_eq!(knowledge.prompts_seen, 2);
        assert!(
            knowledge
                .models
                .iter()
                .all(|m| m.model != "claude-haiku-5-5")
        );
        let opus = find(&knowledge, "claude-opus-5-5");
        assert_eq!(opus.usage[0].efforts.get("xhigh"), Some(&1));
        assert!(
            find(&knowledge, "claude-fable-5-1").usage[0]
                .efforts
                .is_empty()
        );
    }

    #[test]
    fn provider_lists_decide_what_is_offered_and_identity_is_not_retained() {
        let root = scratch("lists");
        std::fs::create_dir_all(root.join("codex")).unwrap();
        std::fs::write(
            root.join("codex/models_cache.json"),
            serde_json::json!({"fetched_at": today(), "models": [
                {"slug": "gpt-6-luna", "display_name": "GPT-6-Luna", "priority": 5, "visibility": "list",
                 "description": "Fast and affordable model for easier tasks.",
                 "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}]},
                {"slug": "hidden-model", "visibility": "hide"}
            ]})
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            root.join("claude.json"),
            serde_json::json!({
                "oauthAccount": {"organizationType": "claude_max", "organizationRateLimitTier": "default_claude_max_20x",
                                 "emailAddress": "person@example.com"},
                "additionalModelOptionsCache": [{"value": "claude-fable-5-1[1m]", "label": "Fable", "description": "Most capable"}]
            })
            .to_string(),
        )
        .unwrap();
        append(
            &root.join("codex/sessions/a.jsonl"),
            &[
                codex_meta(serde_json::json!("vscode")),
                codex_turn("gpt-retired", "low"),
                codex_user(SECRET_PROMPT, "user.text"),
            ],
        );
        let knowledge = learner(&root).refresh();
        let luna = find(&knowledge, "gpt-6-luna");
        assert_eq!(luna.origin, "provider-list");
        assert_eq!(luna.efforts, ["low", "high"]);
        assert!(knowledge.models.iter().all(|m| m.model != "hidden-model"));
        assert!(!find(&knowledge, "gpt-retired").available);
        assert_eq!(find(&knowledge, "claude-fable-5-1").origin, "provider-list");
        let claude = knowledge
            .subscriptions
            .iter()
            .find(|s| s.id == "claude")
            .unwrap();
        assert_eq!(claude.plan.as_deref(), Some("Claude Max 20x"));
        let encoded = serde_json::to_string(&knowledge).unwrap();
        assert!(!encoded.contains("person@example.com"));
        assert!(!encoded.contains("zebra-password-canary"));
    }

    #[test]
    fn recent_use_counts_for_more_than_old_use() {
        let root = scratch("decay");
        let old = serde_json::json!({"timestamp": "2020-01-01T00:00:00Z", "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "hey"}],
            "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["user.text"]}}});
        append(
            &root.join("codex/sessions/a.jsonl"),
            &[
                codex_turn("gpt-old", "low"),
                old,
                codex_turn("gpt-new", "low"),
                codex_user("hey", "user.text"),
            ],
        );
        let knowledge = learner(&root).refresh();
        let old_use = find(&knowledge, "gpt-old").usage[0].weighted;
        let new_use = find(&knowledge, "gpt-new").usage[0].weighted;
        assert!(old_use < 0.01, "{old_use}");
        assert!((new_use - 1.0).abs() < 0.01, "{new_use}");
    }

    #[test]
    fn day_numbers_compare_dates() {
        assert_eq!(
            day_number("2026-03-01").unwrap() - day_number("2026-02-28").unwrap(),
            1
        );
        assert_eq!(
            day_number("2024-03-01").unwrap() - day_number("2024-02-28").unwrap(),
            2
        );
        assert_eq!(day_number("not-a-date"), None);
    }

    #[test]
    fn oversized_lines_are_skipped_and_partial_lines_wait() {
        let root = scratch("lines");
        let path = root.join("a.jsonl");
        let mut body = vec![b'x'; MAX_LINE_BYTES + 10];
        body.push(b'\n');
        body.extend_from_slice(b"short\npartial");
        std::fs::write(&path, &body).unwrap();
        let mut state = FileState::new(Kind::Codex);
        let mut seen = Vec::new();
        read_new_lines(&path, &mut state, |line, _| seen.push(line.to_vec())).unwrap();
        assert_eq!(seen, vec![b"short\n".to_vec()]);
        assert_eq!(state.offset, (body.len() - b"partial".len()) as u64);
    }
}
