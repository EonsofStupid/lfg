//! One assessment of a prompt: intent, complexity, total agents and suggested
//! models, with the formula behind each result.
//!
//! The arithmetic and rules are configuration. Which models exist and how they
//! are used come from [`Knowledge`], which the host rebuilds as the user works.
//! Nothing here keeps prompt text, and scores are rule outputs rather than
//! calibrated probabilities.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::{PreflightRuleConfig, digest_value};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssessmentConfig {
    pub schema_version: String,
    pub revision: String,
    pub max_scored_bytes: usize,
    pub max_task_chars: usize,
    pub word_suffixes: Vec<String>,
    pub task_separators: Vec<String>,
    pub action_words: Vec<String>,
    pub scope_words: Vec<String>,
    /// Words that also indicate an intent, on top of the shared preflight rules.
    pub extra_intent_words: Vec<IntentWords>,
    pub signals: SignalWeights,
    pub brackets: Vec<Bracket>,
    pub roles: Vec<String>,
    pub role_rules: Vec<RoleRule>,
    pub older_words: Vec<String>,
    pub family_ignore_words: Vec<String>,
    pub effort_order: Vec<String>,
    pub matching: ModelMatching,
    pub knowledge_scan: KnowledgeScanLimits,
    pub subscriptions: Vec<SubscriptionDefinition>,
    pub published_models: Vec<PublishedModel>,
}

/// Operator-selected resource limits for local provider history discovery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct KnowledgeScanLimits {
    pub max_directory_entries: usize,
    pub max_files: usize,
    pub max_bytes_per_refresh: usize,
    pub max_line_bytes: usize,
    pub max_pending_per_file: usize,
    pub max_keys_per_file: usize,
    pub max_metadata_bytes: u64,
    pub max_buffered_bytes: usize,
    pub max_identifier_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntentWords {
    pub intent: String,
    pub words: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignalWeights {
    pub work_kinds: f64,
    pub tasks: f64,
    pub length: f64,
    pub words_per_length_point: f64,
    pub length_cap: f64,
    pub scope: f64,
    pub scope_cap: usize,
    pub risks: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bracket {
    pub id: String,
    pub label: String,
    pub plain_label: String,
    pub min_score: f64,
    pub min_agents: usize,
    pub max_agents: usize,
    pub role: String,
    pub effort: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoleRule {
    pub role: String,
    pub words: Vec<String>,
}

/// How a model's fit for the work and the user's own habits combine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelMatching {
    pub usage_influence: f64,
    pub role_step_penalty: f64,
    pub minimum_fit: f64,
    pub unknown_role_fit: f64,
    pub off_ladder_fit: f64,
    pub older_fit_factor: f64,
    pub adjacent_bracket_weight: f64,
    pub usage_half_life_days: f64,
    pub seen_within_days: u32,
    pub max_alternatives: usize,
    pub effort_follows_habit: bool,
    pub refresh_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionDefinition {
    pub id: String,
    pub label: String,
}

/// A published line-up entry, used only where no local source lists the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishedModel {
    pub subscription_id: String,
    pub model: String,
    pub label: String,
    pub description: String,
    pub efforts: Vec<String>,
    pub source: String,
    pub observed_at: String,
}

impl AssessmentConfig {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn default_alpha() -> Self {
        Self::from_json_str(include_str!("../config/assessment.v1.json"))
            .expect("bundled assessment configuration is valid")
    }

    pub fn digest(&self) -> String {
        digest_value(&serde_json::to_value(self).expect("assessment configuration serializes"))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != "lfg.assessment-config.v1"
            || self.revision.trim().is_empty()
            || self.max_scored_bytes == 0
            || self.max_task_chars == 0
        {
            return Err("unsupported assessment schema or empty revision".into());
        }
        let lowercase = |words: &[String]| {
            words
                .iter()
                .all(|word| !word.trim().is_empty() && *word == word.to_lowercase())
        };
        if !lowercase(&self.action_words)
            || !lowercase(&self.scope_words)
            || !lowercase(&self.older_words)
            || !lowercase(&self.family_ignore_words)
            || !lowercase(&self.word_suffixes)
            || self.action_words.is_empty()
            || self
                .extra_intent_words
                .iter()
                .any(|extra| extra.intent.trim().is_empty() || !lowercase(&extra.words))
            || self.task_separators.iter().any(String::is_empty)
        {
            return Err("invalid assessment word list".into());
        }
        let s = &self.signals;
        let weights = [
            s.work_kinds,
            s.tasks,
            s.length,
            s.words_per_length_point,
            s.length_cap,
            s.scope,
            s.risks,
        ];
        if weights.iter().any(|x| !x.is_finite() || *x < 0.0) || s.words_per_length_point == 0.0 {
            return Err("invalid assessment signal weights".into());
        }
        if self.roles.is_empty()
            || self.effort_order.is_empty()
            || self.brackets.is_empty()
            || self.brackets[0].min_score != 0.0
        {
            return Err("assessment needs roles, efforts and a bracket starting at 0".into());
        }
        let mut ids = BTreeSet::new();
        for (index, bracket) in self.brackets.iter().enumerate() {
            let ascending = index == 0 || bracket.min_score > self.brackets[index - 1].min_score;
            if bracket.id.trim().is_empty()
                || !ids.insert(&bracket.id)
                || !ascending
                || !bracket.min_score.is_finite()
                || bracket.min_agents == 0
                || bracket.max_agents < bracket.min_agents
                || !self.roles.contains(&bracket.role)
                || !self.effort_order.contains(&bracket.effort)
            {
                return Err(format!("invalid assessment bracket {}", bracket.id));
            }
        }
        for rule in &self.role_rules {
            if rule.role.trim().is_empty() || rule.words.is_empty() || !lowercase(&rule.words) {
                return Err("invalid assessment role rule".into());
            }
        }
        let m = &self.matching;
        let scan = &self.knowledge_scan;
        if [
            scan.max_directory_entries,
            scan.max_files,
            scan.max_bytes_per_refresh,
            scan.max_line_bytes,
            scan.max_pending_per_file,
            scan.max_keys_per_file,
            scan.max_buffered_bytes,
            scan.max_identifier_bytes,
        ]
        .contains(&0)
            || scan.max_metadata_bytes == 0
            || scan.max_metadata_bytes == u64::MAX
            || scan.max_directory_entries < 2
            || scan.max_files < 2
            || scan
                .max_line_bytes
                .checked_add(1)
                .is_none_or(|n| n > scan.max_bytes_per_refresh)
            || scan
                .max_files
                .checked_mul(scan.max_line_bytes)
                .is_none_or(|bytes| bytes > scan.max_buffered_bytes)
        {
            return Err("invalid knowledge scan limits".into());
        }
        let unit = [
            m.usage_influence,
            m.minimum_fit,
            m.unknown_role_fit,
            m.off_ladder_fit,
            m.older_fit_factor,
            m.adjacent_bracket_weight,
        ];
        if unit
            .iter()
            .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
            || !m.role_step_penalty.is_finite()
            || m.role_step_penalty < 0.0
            || !m.usage_half_life_days.is_finite()
            || m.usage_half_life_days <= 0.0
            || m.refresh_seconds == 0
        {
            return Err("invalid assessment model matching".into());
        }
        for model in &self.published_models {
            if model.subscription_id.trim().is_empty()
                || model.model.trim().is_empty()
                || model.source.trim().is_empty()
                || model.observed_at.trim().is_empty()
            {
                return Err("invalid published model".into());
            }
        }
        Ok(())
    }
}

/// What the host has learned about available models and how they are used.
/// Contains counts and identifiers only, never prompt text.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Knowledge {
    pub state: String,
    pub refreshed_at: Option<String>,
    pub prompts_seen: u64,
    pub sources: Vec<KnowledgeSource>,
    pub subscriptions: Vec<ObservedSubscription>,
    pub models: Vec<KnownModel>,
}

impl Knowledge {
    pub fn disabled() -> Self {
        Self {
            state: "disabled".into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeSource {
    pub id: String,
    pub files: usize,
    pub prompts: u64,
    pub status: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservedSubscription {
    pub id: String,
    pub plan: Option<String>,
    /// When the provider's own model list was last fetched by its local tool.
    pub listed_at: Option<String>,
    pub last_used: Option<String>,
    /// Whole days since `last_used` when this knowledge was refreshed.
    pub days_since_used: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnownModel {
    pub subscription_id: String,
    pub model: String,
    pub label: String,
    pub description: String,
    pub priority: Option<u32>,
    pub efforts: Vec<String>,
    /// `provider-list`, `session` or `published`.
    pub origin: String,
    /// False when a provider list exists and no longer offers this model.
    pub available: bool,
    pub last_used: Option<String>,
    pub usage: Vec<BracketUsage>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BracketUsage {
    pub bracket: String,
    pub prompts: u64,
    /// Prompt count discounted by age, so recent work counts for more.
    pub weighted: f64,
    pub efforts: BTreeMap<String, u64>,
}

/// The result of an explicit sign-in check, such as a provider CLI probe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionCheck {
    pub active: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssessRequest {
    pub session_id: String,
    pub revision: u64,
    pub text: String,
    /// Subscriptions the user says they have. These stay marked as declared.
    #[serde(default)]
    pub declared_subscriptions: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assessment {
    pub schema_version: String,
    pub session_id: String,
    pub revision: u64,
    pub duration_micros: u64,
    pub config_revision: String,
    pub config_digest: String,
    pub prompt: PromptFacts,
    pub intent: IntentResult,
    pub complexity: ComplexityResult,
    pub agents: AgentResult,
    pub models: ModelResult,
    pub subscriptions: Vec<SubscriptionStatus>,
    pub knowledge: KnowledgeSummary,
    /// Identifiers of what fired, safe to retain: no prompt words.
    pub rules: Vec<String>,
    pub advisory: bool,
    pub confidence: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFacts {
    pub characters: usize,
    pub words: usize,
    pub tasks: Vec<String>,
    pub scoring_truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordHit {
    pub id: String,
    pub plain_label: String,
    pub matched: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntentResult {
    pub primary: String,
    pub plain_label: String,
    pub all: Vec<WordHit>,
    pub risks: Vec<WordHit>,
    /// True when no rule matched and the general fallback was used.
    pub fallback: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormulaTerm {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub weight: f64,
    pub points: f64,
    pub detail: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LadderStep {
    pub id: String,
    pub label: String,
    pub min_score: f64,
    pub current: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComplexityResult {
    pub score: f64,
    pub bracket: String,
    pub label: String,
    pub plain_label: String,
    pub terms: Vec<FormulaTerm>,
    pub ladder: Vec<LadderStep>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentResult {
    pub total: usize,
    pub tasks: usize,
    pub min: usize,
    pub max: usize,
    pub plain: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSuggestion {
    pub subscription_id: String,
    pub model: String,
    pub label: String,
    pub description: String,
    pub role: Option<String>,
    pub effort: Option<String>,
    pub usual_effort: Option<String>,
    /// `provider-list`, `session` or `published`.
    pub origin: String,
    pub fit: f64,
    pub habit: f64,
    pub score: f64,
    pub prompts_in_bracket: u64,
    pub why: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelResult {
    pub role: String,
    pub effort: String,
    pub suggested: Option<ModelSuggestion>,
    pub alternatives: Vec<ModelSuggestion>,
    pub usual_model: Option<String>,
    pub plain: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionStatus {
    pub id: String,
    pub label: String,
    /// `verified`, `recognized`, `declared`, `signed-out` or `not-found`.
    pub status: String,
    pub usable: bool,
    pub declared: bool,
    pub plan: Option<String>,
    pub last_used: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeSummary {
    pub state: String,
    pub refreshed_at: Option<String>,
    pub prompts_seen: u64,
    pub models_known: usize,
}

/// True when `word` appears as a whole word or phrase, allowing a configured
/// ending such as `tests` for `test`, or `updating` for `update`.
pub fn contains_word(text: &str, word: &str, suffixes: &[String]) -> bool {
    let ends_cleanly = |rest: &str, allowed: &dyn Fn(&str) -> bool| {
        let token_end = rest
            .find(|c: char| !c.is_alphanumeric())
            .unwrap_or(rest.len());
        let token = &rest[..token_end];
        token.is_empty() || allowed(token)
    };
    let starts_cleanly = |start: usize| {
        !text[..start]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
    };
    let search = |needle: &str, allowed: &dyn Fn(&str) -> bool| {
        text.match_indices(needle).any(|(start, _)| {
            starts_cleanly(start) && ends_cleanly(&text[start + needle.len()..], allowed)
        })
    };
    if search(word, &|token| suffixes.iter().any(|s| s == token)) {
        return true;
    }
    match word.strip_suffix('e') {
        Some(stem) if !stem.is_empty() && suffixes.iter().any(|s| s == "ing") => {
            text.match_indices(stem).any(|(start, _)| {
                starts_cleanly(start) && {
                    let rest = &text[start + stem.len()..];
                    rest.strip_prefix("ing")
                        .is_some_and(|after| ends_cleanly(after, &|_| false))
                }
            })
        }
        _ => false,
    }
}

fn matched_words(text: &str, words: &[String], suffixes: &[String]) -> Vec<String> {
    words
        .iter()
        .filter(|word| contains_word(text, word, suffixes))
        .cloned()
        .collect()
}

fn scored_slice(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn task_clauses(normalized: &str, config: &AssessmentConfig) -> Vec<String> {
    let mut clauses = vec![normalized.to_string()];
    for separator in &config.task_separators {
        clauses = clauses
            .iter()
            .flat_map(|clause| clause.split(separator.as_str()))
            .map(str::to_string)
            .collect();
    }
    clauses
        .into_iter()
        .map(|clause| clause.trim().trim_start_matches("and ").trim().to_string())
        .filter(|clause| {
            config
                .action_words
                .iter()
                .any(|word| contains_word(clause, word, &config.word_suffixes))
        })
        .map(|clause| {
            if clause.chars().count() > config.max_task_chars {
                let short: String = clause.chars().take(config.max_task_chars).collect();
                format!("{short}…")
            } else {
                clause
            }
        })
        .collect()
}

struct Scored {
    characters: usize,
    words: usize,
    tasks: Vec<String>,
    intents: Vec<WordHit>,
    fallback: bool,
    risks: Vec<WordHit>,
    terms: Vec<FormulaTerm>,
    score: f64,
    bracket: usize,
}

fn round2(value: f64) -> f64 {
    // Adding zero turns the -0.0 of an empty float sum into 0.0.
    (value * 100.0).round() / 100.0 + 0.0
}

fn score_prompt(config: &AssessmentConfig, rules: &PreflightRuleConfig, text: &str) -> Scored {
    let normalized = scored_slice(text, config.max_scored_bytes).to_lowercase();
    let suffixes = &config.word_suffixes;
    let words = text.split_whitespace().count();
    let intents: Vec<WordHit> = rules
        .rules
        .iter()
        .filter_map(|rule| {
            let mut matched = matched_words(&normalized, &rule.keywords, suffixes);
            for extra in config
                .extra_intent_words
                .iter()
                .filter(|extra| extra.intent == rule.intent)
            {
                matched.extend(matched_words(&normalized, &extra.words, suffixes));
            }
            (!matched.is_empty()).then(|| WordHit {
                id: rule.intent.clone(),
                plain_label: rule.plain_label.clone(),
                matched,
            })
        })
        .collect();
    let risks: Vec<WordHit> = rules
        .risk_rules
        .iter()
        .filter_map(|rule| {
            let matched = matched_words(&normalized, &rule.keywords, suffixes);
            (!matched.is_empty()).then(|| WordHit {
                id: rule.risk.clone(),
                plain_label: rule.plain_label.clone(),
                matched,
            })
        })
        .collect();
    let tasks = task_clauses(&normalized, config);
    let scope = matched_words(&normalized, &config.scope_words, suffixes);
    let s = &config.signals;
    let term = |id: &str, label: &str, value: f64, weight: f64, detail: Vec<String>| FormulaTerm {
        id: id.into(),
        label: label.into(),
        value: round2(value),
        weight,
        points: value * weight,
        detail,
    };
    let terms = vec![
        term(
            "work_kinds",
            "Kinds of work",
            intents.len() as f64,
            s.work_kinds,
            intents.iter().map(|hit| hit.plain_label.clone()).collect(),
        ),
        term(
            "tasks",
            "Separate tasks",
            tasks.len() as f64,
            s.tasks,
            Vec::new(),
        ),
        term(
            "length",
            "Length",
            (words as f64 / s.words_per_length_point).min(s.length_cap),
            s.length,
            vec![format!("{words} words")],
        ),
        term(
            "scope",
            "Wide-reaching words",
            scope.len().min(s.scope_cap) as f64,
            s.scope,
            scope,
        ),
        term(
            "risks",
            "Sensitive content",
            risks.len() as f64,
            s.risks,
            risks.iter().map(|hit| hit.plain_label.clone()).collect(),
        ),
    ];
    let score: f64 = terms.iter().map(|term| term.points).sum();
    let bracket = config
        .brackets
        .iter()
        .rposition(|bracket| score >= bracket.min_score)
        .unwrap_or(0);
    let fallback = intents.is_empty();
    let terms = terms
        .into_iter()
        .map(|term| FormulaTerm {
            points: round2(term.points),
            ..term
        })
        .collect();
    Scored {
        characters: text.chars().count(),
        words,
        tasks,
        intents,
        fallback,
        risks,
        terms,
        score,
        bracket,
    }
}

/// The complexity bracket for a prompt, using the same formula as [`assess`].
pub fn bracket_for(config: &AssessmentConfig, rules: &PreflightRuleConfig, text: &str) -> usize {
    score_prompt(config, rules, text).bracket
}

fn described_role(config: &AssessmentConfig, description: &str) -> Option<String> {
    let text = description.to_lowercase();
    config
        .role_rules
        .iter()
        .find(|rule| {
            rule.words
                .iter()
                .any(|word| contains_word(&text, word, &config.word_suffixes))
        })
        .map(|rule| rule.role.clone())
}

fn family_words<'a>(config: &AssessmentConfig, model: &'a str) -> Vec<&'a str> {
    model
        .split('-')
        .filter(|word| {
            word.chars().any(char::is_alphabetic)
                && !config.family_ignore_words.iter().any(|skip| skip == word)
        })
        .collect()
}

/// Everything the assessment can rank: what the host learned, completed from the
/// published line-up, with each model's role read from its own description or,
/// failing that, inherited from a sibling in the same family.
pub fn known_models(config: &AssessmentConfig, knowledge: &Knowledge) -> Vec<KnownModel> {
    let mut models = knowledge.models.clone();
    for published in &config.published_models {
        match models
            .iter_mut()
            .find(|m| m.subscription_id == published.subscription_id && m.model == published.model)
        {
            Some(known) => {
                if known.description.is_empty() {
                    known.description = published.description.clone();
                }
                if known.label.is_empty() || known.label == known.model {
                    known.label = published.label.clone();
                }
                if known.efforts.is_empty() {
                    known.efforts = published.efforts.clone();
                }
            }
            None => models.push(KnownModel {
                subscription_id: published.subscription_id.clone(),
                model: published.model.clone(),
                label: published.label.clone(),
                description: published.description.clone(),
                efforts: published.efforts.clone(),
                origin: "published".into(),
                available: true,
                ..KnownModel::default()
            }),
        }
    }
    let direct: Vec<Option<String>> = models
        .iter()
        .map(|model| described_role(config, &model.description))
        .collect();
    let roles: Vec<Option<String>> = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            direct[index].clone().or_else(|| {
                let family = family_words(config, &model.model);
                models
                    .iter()
                    .enumerate()
                    .filter(|(other, sibling)| {
                        *other != index
                            && sibling.subscription_id == model.subscription_id
                            && direct[*other].is_some()
                            && family_words(config, &sibling.model)
                                .iter()
                                .any(|word| family.contains(word))
                    })
                    .min_by_key(|(_, sibling)| sibling.priority.unwrap_or(u32::MAX))
                    .and_then(|(other, _)| direct[other].clone())
            })
        })
        .collect();
    for (model, role) in models.iter_mut().zip(roles) {
        model.role = role;
    }
    models
}

fn fit_for(config: &AssessmentConfig, model: &KnownModel, target_role: &str) -> f64 {
    let m = &config.matching;
    let target = config
        .roles
        .iter()
        .position(|role| role == target_role)
        .unwrap_or(0);
    let base = match &model.role {
        None => m.unknown_role_fit,
        Some(role) => match config.roles.iter().position(|known| known == role) {
            None => m.off_ladder_fit,
            Some(position) => {
                (1.0 - m.role_step_penalty * position.abs_diff(target) as f64).max(m.minimum_fit)
            }
        },
    };
    let description = model.description.to_lowercase();
    let older = config
        .older_words
        .iter()
        .any(|word| contains_word(&description, word, &config.word_suffixes));
    if older {
        base * m.older_fit_factor
    } else {
        base
    }
}

fn weighted_near(config: &AssessmentConfig, model: &KnownModel, bracket: usize) -> f64 {
    let adjacent = config.matching.adjacent_bracket_weight;
    model
        .usage
        .iter()
        .filter_map(|usage| {
            let position = config.brackets.iter().position(|b| b.id == usage.bracket)?;
            match position.abs_diff(bracket) {
                0 => Some(usage.weighted),
                1 => Some(usage.weighted * adjacent),
                _ => None,
            }
        })
        .sum()
}

fn usual_effort(model: &KnownModel, bracket_id: &str) -> Option<String> {
    model
        .usage
        .iter()
        .find(|usage| usage.bracket == bracket_id)
        .and_then(|usage| {
            usage
                .efforts
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(effort, _)| effort.clone())
        })
}

/// The target effort if this model supports it, otherwise the nearest level
/// below it, otherwise the model's lowest level.
fn effort_for(config: &AssessmentConfig, model: &KnownModel, target: &str) -> Option<String> {
    if model.efforts.is_empty() {
        return None;
    }
    let rank = |effort: &str| config.effort_order.iter().position(|e| e == effort);
    let target_rank = rank(target)?;
    model
        .efforts
        .iter()
        .filter_map(|effort| rank(effort).map(|position| (position, effort)))
        .filter(|(position, _)| *position <= target_rank)
        .max_by_key(|(position, _)| *position)
        .or_else(|| {
            model
                .efforts
                .iter()
                .filter_map(|effort| rank(effort).map(|position| (position, effort)))
                .min_by_key(|(position, _)| *position)
        })
        .map(|(_, effort)| effort.clone())
}

fn rank_models(
    config: &AssessmentConfig,
    models: &[KnownModel],
    usable: &BTreeSet<String>,
    bracket_index: usize,
) -> Vec<ModelSuggestion> {
    let bracket = &config.brackets[bracket_index];
    let candidates: Vec<&KnownModel> = models
        .iter()
        .filter(|model| model.available && usable.contains(&model.subscription_id))
        .collect();
    let total: f64 = candidates
        .iter()
        .map(|model| weighted_near(config, model, bracket_index))
        .sum();
    let bracket_total: u64 = candidates
        .iter()
        .map(|model| prompts_in(model, &bracket.id))
        .sum();
    let influence = config.matching.usage_influence;
    let mut ranked: Vec<(ModelSuggestion, Option<u32>)> = candidates
        .iter()
        .map(|model| {
            let fit = fit_for(config, model, &bracket.role);
            let habit = if total > 0.0 {
                weighted_near(config, model, bracket_index) / total
            } else {
                0.0
            };
            let score = (1.0 - influence) * fit + influence * habit;
            let usual = usual_effort(model, &bracket.id);
            let effort = if config.matching.effort_follows_habit {
                usual
                    .clone()
                    .or_else(|| effort_for(config, model, &bracket.effort))
            } else {
                effort_for(config, model, &bracket.effort)
            };
            let prompts = prompts_in(model, &bracket.id);
            let role_text = model.role.as_deref().unwrap_or("unclassified");
            let usage_text = if prompts > 0 {
                let at = usual
                    .as_deref()
                    .map(|e| format!(", usually at {e} effort"))
                    .unwrap_or_default();
                format!(
                    "You used it for {prompts} of {bracket_total} {} requests{at}.",
                    bracket.label.to_lowercase()
                )
            } else {
                format!(
                    "You haven't used it for {} requests yet.",
                    bracket.label.to_lowercase()
                )
            };
            let why = format!("A {role_text} model. {usage_text}");
            (
                ModelSuggestion {
                    subscription_id: model.subscription_id.clone(),
                    model: model.model.clone(),
                    label: if model.label.is_empty() {
                        model.model.clone()
                    } else {
                        model.label.clone()
                    },
                    description: model.description.clone(),
                    role: model.role.clone(),
                    effort,
                    usual_effort: usual,
                    origin: model.origin.clone(),
                    fit: round2(fit),
                    habit: round2(habit),
                    score: round2(score),
                    prompts_in_bracket: prompts,
                    why,
                },
                model.priority,
            )
        })
        .collect();
    ranked.sort_by(|(a, a_priority), (b, b_priority)| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| {
                a_priority
                    .unwrap_or(u32::MAX)
                    .cmp(&b_priority.unwrap_or(u32::MAX))
            })
            .then_with(|| a.model.cmp(&b.model))
    });
    ranked.into_iter().map(|(model, _)| model).collect()
}

fn prompts_in(model: &KnownModel, bracket_id: &str) -> u64 {
    model
        .usage
        .iter()
        .find(|usage| usage.bracket == bracket_id)
        .map_or(0, |usage| usage.prompts)
}

fn subscription_statuses(
    config: &AssessmentConfig,
    request: &AssessRequest,
    knowledge: &Knowledge,
    checks: &BTreeMap<String, SubscriptionCheck>,
) -> Vec<SubscriptionStatus> {
    let mut ids: BTreeSet<&str> = config.subscriptions.iter().map(|s| s.id.as_str()).collect();
    ids.extend(knowledge.subscriptions.iter().map(|s| s.id.as_str()));
    ids.extend(request.declared_subscriptions.iter().map(String::as_str));
    ids.into_iter()
        .map(|id| {
            let label = config
                .subscriptions
                .iter()
                .find(|s| s.id == id)
                .map_or(id, |s| s.label.as_str());
            let observed = knowledge.subscriptions.iter().find(|s| s.id == id);
            let declared = request.declared_subscriptions.iter().any(|d| d == id);
            let check = checks.get(id);
            let plan = observed.and_then(|s| s.plan.clone());
            let last_used = observed.and_then(|s| s.last_used.clone());
            let recently_used = observed
                .and_then(|s| s.days_since_used)
                .is_some_and(|days| days <= config.matching.seen_within_days);
            let listed = observed.is_some_and(|s| s.listed_at.is_some());
            let recognized = plan.is_some() || recently_used || listed;
            let used_text = last_used
                .as_deref()
                .map(|day| format!("last used {day}"))
                .unwrap_or_else(|| "no recent sessions".into());
            let (status, detail) = match check {
                Some(check) if check.active => (
                    "signed-in",
                    format!("Signed in; subscription entitlement is not verified; {used_text}."),
                ),
                Some(check) if !declared => ("signed-out", check.detail.clone()),
                _ if recognized => ("recognized", format!("Found on this machine; {used_text}.")),
                _ if declared => (
                    "declared",
                    "You said you have this; LFG could not confirm it.".to_string(),
                ),
                _ => ("not-found", "Not found on this machine.".to_string()),
            };
            SubscriptionStatus {
                id: id.to_string(),
                label: label.to_string(),
                status: status.into(),
                usable: declared,
                declared,
                plan,
                last_used,
                detail,
            }
        })
        .collect()
}

/// Every model the host knows about with its role resolved, for inspection.
pub fn with_roles(config: &AssessmentConfig, knowledge: &Knowledge) -> Knowledge {
    Knowledge {
        models: known_models(config, knowledge),
        ..knowledge.clone()
    }
}

pub fn assess(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &AssessRequest,
    knowledge: &Knowledge,
    checks: &BTreeMap<String, SubscriptionCheck>,
) -> Assessment {
    let started = Instant::now();
    let scored = score_prompt(config, rules, &request.text);
    let bracket = &config.brackets[scored.bracket];

    let tasks = scored.tasks.len();
    let total = tasks.max(1).clamp(bracket.min_agents, bracket.max_agents);
    let range = if bracket.min_agents == bracket.max_agents {
        format!("{}", bracket.min_agents)
    } else {
        format!("{}–{}", bracket.min_agents, bracket.max_agents)
    };
    let agents_plain = format!(
        "{tasks} separate task{} found. {} work uses {range} agent{}, so LFG suggests {total}.",
        if tasks == 1 { "" } else { "s" },
        bracket.label,
        if bracket.max_agents == 1 { "" } else { "s" },
    );

    let subscriptions = subscription_statuses(config, request, knowledge, checks);
    let usable: BTreeSet<String> = subscriptions
        .iter()
        .filter(|s| s.usable)
        .map(|s| s.id.clone())
        .collect();
    let models = known_models(config, knowledge);
    let mut ranked = rank_models(config, &models, &usable, scored.bracket);
    let usual_model = ranked
        .iter()
        .filter(|model| model.prompts_in_bracket > 0)
        .max_by_key(|model| model.prompts_in_bracket)
        .map(|model| model.label.clone());
    let suggested = (!ranked.is_empty()).then(|| ranked.remove(0));
    ranked.truncate(config.matching.max_alternatives);
    let models_plain = match &suggested {
        None if usable.is_empty() => {
            "LFG found no subscriptions yet. Mark one below to get a suggestion.".to_string()
        }
        None => "LFG knows no models for your subscriptions yet.".to_string(),
        Some(model) => {
            let mut text = format!(
                "{} work suits a {} model at {} effort. ",
                bracket.label, bracket.role, bracket.effort
            );
            if model.role.as_deref() == Some(bracket.role.as_str()) {
                text.push_str(&format!("{} is the best match.", model.label));
            } else if model.prompts_in_bracket > 0 {
                text.push_str(&format!(
                    "{} ranks first because it is close to that and you use it most for this work.",
                    model.label
                ));
            } else {
                text.push_str(&format!("{} is the closest match you have.", model.label));
            }
            if let Some(usual) = usual_model.as_ref().filter(|usual| **usual != model.label) {
                text.push_str(&format!(" You usually use {usual} for this kind of work."));
            }
            text
        }
    };

    let mut fired: Vec<String> = scored
        .intents
        .iter()
        .map(|hit| format!("intent:{}", hit.id))
        .collect();
    if scored.fallback {
        fired.push(format!("intent-fallback:{}", rules.fallback.intent));
    }
    fired.extend(scored.risks.iter().map(|hit| format!("risk:{}", hit.id)));
    fired.extend(
        scored
            .terms
            .iter()
            .map(|term| format!("points:{}={}", term.id, term.points)),
    );
    fired.push(format!("bracket:{}", bracket.id));
    fired.push(format!("agents:{total}"));
    fired.push(format!("role:{}", bracket.role));
    if let Some(model) = &suggested {
        fired.push(format!("model:{}/{}", model.subscription_id, model.model));
        fired.push(format!("model-origin:{}", model.origin));
        if let Some(effort) = &model.effort {
            fired.push(format!("effort:{effort}"));
        }
    }

    let (primary, plain_label) = scored.intents.first().map_or_else(
        || {
            (
                rules.fallback.intent.clone(),
                rules.fallback.plain_label.clone(),
            )
        },
        |hit| (hit.id.clone(), hit.plain_label.clone()),
    );

    Assessment {
        schema_version: "lfg.assessment.v1".into(),
        session_id: request.session_id.clone(),
        revision: request.revision,
        duration_micros: u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        config_revision: config.revision.clone(),
        config_digest: config.digest(),
        prompt: PromptFacts {
            characters: scored.characters,
            words: scored.words,
            tasks: scored.tasks,
            scoring_truncated: request.text.len() > config.max_scored_bytes,
        },
        intent: IntentResult {
            primary,
            plain_label,
            all: scored.intents,
            risks: scored.risks,
            fallback: scored.fallback,
        },
        complexity: ComplexityResult {
            score: round2(scored.score),
            bracket: bracket.id.clone(),
            label: bracket.label.clone(),
            plain_label: bracket.plain_label.clone(),
            terms: scored.terms,
            ladder: config
                .brackets
                .iter()
                .enumerate()
                .map(|(index, step)| LadderStep {
                    id: step.id.clone(),
                    label: step.label.clone(),
                    min_score: step.min_score,
                    current: index == scored.bracket,
                })
                .collect(),
        },
        agents: AgentResult {
            total,
            tasks,
            min: bracket.min_agents,
            max: bracket.max_agents,
            plain: agents_plain,
        },
        models: ModelResult {
            role: bracket.role.clone(),
            effort: bracket.effort.clone(),
            suggested,
            alternatives: ranked,
            usual_model,
            plain: models_plain,
        },
        subscriptions,
        knowledge: KnowledgeSummary {
            state: knowledge.state.clone(),
            refreshed_at: knowledge.refreshed_at.clone(),
            prompts_seen: knowledge.prompts_seen,
            models_known: models.len(),
        },
        rules: fired,
        advisory: true,
        confidence: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> PreflightRuleConfig {
        PreflightRuleConfig::from_json_str(include_str!("../config/preflight.v1.json")).unwrap()
    }

    fn request(text: &str, declared: &[&str]) -> AssessRequest {
        AssessRequest {
            session_id: "test".into(),
            revision: 1,
            text: text.into(),
            declared_subscriptions: declared.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn model(subscription: &str, name: &str, description: &str, priority: u32) -> KnownModel {
        KnownModel {
            subscription_id: subscription.into(),
            model: name.into(),
            label: name.into(),
            description: description.into(),
            priority: Some(priority),
            efforts: vec!["low".into(), "medium".into(), "high".into(), "xhigh".into()],
            origin: "provider-list".into(),
            available: true,
            ..KnownModel::default()
        }
    }

    fn used(mut known: KnownModel, bracket: &str, prompts: u64, effort: &str) -> KnownModel {
        known.usage.push(BracketUsage {
            bracket: bracket.into(),
            prompts,
            weighted: prompts as f64,
            efforts: BTreeMap::from([(effort.to_string(), prompts)]),
        });
        known
    }

    fn knowledge(models: Vec<KnownModel>) -> Knowledge {
        Knowledge {
            state: "ready".into(),
            subscriptions: vec![ObservedSubscription {
                id: "codex".into(),
                listed_at: Some("2026-11-01".into()),
                ..ObservedSubscription::default()
            }],
            models,
            ..Knowledge::default()
        }
    }

    const MULTI: &str = "Fix the auth token refresh bug in the Rust server, add regression tests, and update the docs";

    fn configured_fixture_models() -> AssessmentConfig {
        let mut config = AssessmentConfig::default_alpha();
        config.published_models = vec![
            PublishedModel {
                subscription_id: "claude".into(),
                model: "fixture-frontier".into(),
                label: "Fixture frontier".into(),
                description: "frontier".into(),
                efforts: vec!["low".into(), "high".into()],
                source: "test".into(),
                observed_at: "fixture".into(),
            },
            PublishedModel {
                subscription_id: "claude".into(),
                model: "fixture-fast".into(),
                label: "Fixture fast".into(),
                description: "fast".into(),
                efforts: vec!["low".into(), "high".into()],
                source: "test".into(),
                observed_at: "fixture".into(),
            },
        ];
        config
    }

    #[test]
    fn bundled_configuration_is_valid() {
        let config = AssessmentConfig::default_alpha();
        assert_eq!(config.brackets[0].min_score, 0.0);
        assert_eq!(config.digest().len(), 64);
    }

    #[test]
    fn words_match_whole_words_with_configured_endings() {
        let suffixes = AssessmentConfig::default_alpha().word_suffixes;
        for (text, word) in [
            ("run the tests now", "test"),
            ("testing it", "test"),
            ("updating the docs", "update"),
            ("updated docs", "update"),
            ("set up the server", "set up"),
        ] {
            assert!(contains_word(text, word, &suffixes), "{text} / {word}");
        }
        for (text, word) in [
            ("the latest news", "test"),
            ("a prefix here", "fix"),
            ("tokenizer settings", "token"),
            ("contest", "test"),
        ] {
            assert!(!contains_word(text, word, &suffixes), "{text} / {word}");
        }
    }

    #[test]
    fn all_four_results_come_from_one_formula() {
        let config = configured_fixture_models();
        let result = assess(
            &config,
            &rules(),
            &request(MULTI, &["claude"]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
        );
        assert_eq!(result.intent.primary, "build.software");
        assert_eq!(result.intent.all.len(), 2);
        assert!(result.intent.all[0].matched.contains(&"add".to_string()));
        assert_eq!(result.prompt.tasks.len(), 3);
        assert_eq!(result.complexity.bracket, "complex");
        assert_eq!(result.agents.total, 3);
        let points: f64 = result.complexity.terms.iter().map(|t| t.points).sum();
        assert!((points - result.complexity.score).abs() < 0.03);
        assert_eq!(
            result
                .complexity
                .ladder
                .iter()
                .filter(|s| s.current)
                .count(),
            1
        );
        assert_eq!(result.models.role, "frontier");
        let suggested = result.models.suggested.unwrap();
        assert_eq!(suggested.model, "fixture-frontier");
        assert_eq!(suggested.origin, "published");
        assert_eq!(suggested.effort.as_deref(), Some("high"));
        assert!(result.rules.contains(&"bracket:complex".to_string()));
        assert!(result.rules.iter().all(|rule| !rule.contains("refresh")));
    }

    #[test]
    fn simple_requests_stay_small() {
        let config = configured_fixture_models();
        for text in ["hey thanks", "What is a closure in Rust?"] {
            let result = assess(
                &config,
                &rules(),
                &request(text, &["claude"]),
                &Knowledge::disabled(),
                &BTreeMap::new(),
            );
            assert_eq!(result.complexity.bracket, "trivial", "{text}");
            assert_eq!(result.agents.total, 1);
            assert_eq!(result.intent.primary, "respond.general");
            assert!(result.intent.fallback);
            let suggested = result.models.suggested.unwrap();
            assert_eq!(suggested.model, "fixture-fast");
            assert_eq!(suggested.effort.as_deref(), Some("low"));
        }
    }

    #[test]
    fn intent_words_extend_the_shared_rules() {
        let config = AssessmentConfig::default_alpha();
        let result = assess(
            &config,
            &rules(),
            &request("Refactor the payment module and deploy to production", &[]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
        );
        assert_eq!(result.intent.primary, "build.software");
        assert!(result.models.suggested.is_none());
        assert_eq!(
            serde_json::to_value(round2(-0.0)).unwrap().to_string(),
            "0.0"
        );
    }

    #[test]
    fn roles_come_from_provider_descriptions_and_siblings() {
        let config = AssessmentConfig::default_alpha();
        let models = known_models(
            &config,
            &knowledge(vec![
                model(
                    "codex",
                    "gpt-6-luna",
                    "Fast and affordable model for easier tasks.",
                    5,
                ),
                model(
                    "codex",
                    "gpt-6-astra",
                    "Frontier intelligence for the most demanding work.",
                    2,
                ),
                model(
                    "codex",
                    "gpt-5.6-terra",
                    "Older balanced model for straightforward work.",
                    8,
                ),
                model("codex", "gpt-6.1-terra", "GPT-6.1 Terra Codex model.", 3),
                model(
                    "codex",
                    "gpt-daybreak-blue-latest",
                    "Latest frontier agentic coding model for broad defensive cybersecurity work.",
                    11,
                ),
                model("codex", "gpt-mystery", "A new model.", 12),
            ]),
        );
        let role = |name: &str| {
            models
                .iter()
                .find(|m| m.model == name)
                .and_then(|m| m.role.clone())
        };
        assert_eq!(role("gpt-6-luna").as_deref(), Some("fast"));
        assert_eq!(role("gpt-6-astra").as_deref(), Some("frontier"));
        assert_eq!(role("gpt-6.1-terra").as_deref(), Some("standard"));
        assert_eq!(
            role("gpt-daybreak-blue-latest").as_deref(),
            Some("specialist")
        );
        assert_eq!(role("gpt-mystery"), None);
    }

    #[test]
    fn suggestions_blend_fit_with_the_users_habits() {
        let config = AssessmentConfig::default_alpha();
        let codex = |models: Vec<KnownModel>| {
            assess(
                &config,
                &rules(),
                &request("hey thanks", &["codex"]),
                &knowledge(models),
                &BTreeMap::new(),
            )
            .models
        };
        // A fast model still wins trivial work over a heavily used frontier model.
        let result = codex(vec![
            used(
                model(
                    "codex",
                    "gpt-6-astra",
                    "Frontier intelligence for the most demanding work.",
                    2,
                ),
                "trivial",
                100,
                "xhigh",
            ),
            model(
                "codex",
                "gpt-6-luna",
                "Fast and affordable model for easier tasks.",
                5,
            ),
        ]);
        assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-6-luna");
        assert_eq!(result.usual_model.as_deref(), Some("gpt-6-astra"));
        assert!(result.plain.contains("You usually use gpt-6-astra"));

        // Between two models of the same role, the one the user actually uses wins.
        let result = codex(vec![
            model(
                "codex",
                "gpt-6-luna",
                "Fast and affordable model for easier tasks.",
                5,
            ),
            used(
                model("codex", "gpt-7-luna", "Fast model for easier tasks.", 9),
                "trivial",
                5,
                "low",
            ),
        ]);
        assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-7-luna");

        // Without usage, the provider's own ordering breaks the tie.
        let result = codex(vec![
            model("codex", "gpt-7-luna", "Fast model for easier tasks.", 9),
            model(
                "codex",
                "gpt-6-luna",
                "Fast and affordable model for easier tasks.",
                5,
            ),
        ]);
        assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-6-luna");
    }

    #[test]
    fn retired_and_unusable_models_are_never_suggested() {
        let config = AssessmentConfig::default_alpha();
        let mut retired = used(
            model("codex", "gpt-old-luna", "Fast model.", 1),
            "trivial",
            500,
            "low",
        );
        retired.available = false;
        let result = assess(
            &config,
            &rules(),
            &request("hey thanks", &["codex"]),
            &knowledge(vec![
                retired,
                model("codex", "gpt-6-luna", "Fast model.", 5),
                model("other", "other-fast", "Fast model.", 1),
            ]),
            &BTreeMap::new(),
        );
        assert_eq!(result.models.suggested.unwrap().model, "gpt-6-luna");
        assert!(
            result
                .models
                .alternatives
                .iter()
                .all(|m| m.subscription_id == "codex")
        );
    }

    #[test]
    fn subscription_status_reflects_evidence() {
        let config = AssessmentConfig::default_alpha();
        let observed = Knowledge {
            subscriptions: vec![
                ObservedSubscription {
                    id: "claude".into(),
                    plan: Some("Claude Max 20x".into()),
                    ..ObservedSubscription::default()
                },
                ObservedSubscription {
                    id: "stale".into(),
                    last_used: Some("2020-01-01".into()),
                    days_since_used: Some(2000),
                    ..ObservedSubscription::default()
                },
            ],
            ..Knowledge::default()
        };
        let checks = BTreeMap::from([(
            "codex".to_string(),
            SubscriptionCheck {
                active: false,
                detail: "Codex is not signed in".into(),
            },
        )]);
        let result = assess(
            &config,
            &rules(),
            &request("hey", &["mine"]),
            &observed,
            &checks,
        );
        let status = |id: &str| {
            result
                .subscriptions
                .iter()
                .find(|s| s.id == id)
                .map(|s| (s.status.as_str(), s.usable))
                .unwrap()
        };
        assert_eq!(status("claude"), ("recognized", false));
        assert_eq!(status("codex"), ("signed-out", false));
        assert_eq!(status("mine"), ("declared", true));
        assert_eq!(status("stale"), ("not-found", false));
    }

    #[test]
    fn remembered_use_does_not_enable_an_undeclared_subscription() {
        let result = assess(
            &AssessmentConfig::default_alpha(),
            &rules(),
            &request("hey", &[]),
            &knowledge(vec![model("codex", "listed-model", "Fast model.", 1)]),
            &BTreeMap::new(),
        );
        assert!(result.models.suggested.is_none());
        let codex = result
            .subscriptions
            .iter()
            .find(|item| item.id == "codex")
            .unwrap();
        assert_eq!(codex.status, "recognized");
        assert!(!codex.usable);
    }

    #[test]
    fn effort_uses_the_nearest_supported_level() {
        let config = AssessmentConfig::default_alpha();
        let mut known = model("codex", "m", "Fast model.", 1);
        known.efforts = vec!["low".into(), "high".into()];
        assert_eq!(
            effort_for(&config, &known, "medium").as_deref(),
            Some("low")
        );
        assert_eq!(
            effort_for(&config, &known, "xhigh").as_deref(),
            Some("high")
        );
        known.efforts = vec!["high".into()];
        assert_eq!(effort_for(&config, &known, "low").as_deref(), Some("high"));
        known.efforts = vec![];
        assert_eq!(effort_for(&config, &known, "low"), None);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/assessment.v1.json")).unwrap();
        value["brackets"][1]["minScore"] = serde_json::json!(0.0);
        assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/assessment.v1.json")).unwrap();
        value["knowledgeScan"]["maxBytesPerRefresh"] = serde_json::json!(1);
        assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../config/assessment.v1.json")).unwrap();
        value["maxScoredBytes"] = serde_json::json!(0);
        assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
    }

    #[test]
    fn scoring_limit_is_configurable_and_reported() {
        let mut config = AssessmentConfig::default_alpha();
        config.max_scored_bytes = 4;
        let result = assess(
            &config,
            &rules(),
            &request("fix the system", &[]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
        );
        assert!(result.prompt.scoring_truncated);
    }
}
