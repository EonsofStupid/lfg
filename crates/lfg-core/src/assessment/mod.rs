//! One assessment of a prompt: intent, complexity, total agents and suggested
//! models, with the formula behind each result.
//!
//! Each job has its own module, and [`assess_with_model`] runs them in order:
//!
//! 1. [`intent`]: what kind of work, and any sensitive content.
//! 2. [`complexity`]: the points formula and its bracket, or the trained
//!    model's bracket when one is bound.
//! 3. [`agents`]: how many agents, within the chosen bracket's range.
//! 4. [`models`] and [`subscriptions`]: which model suits the bracket's role,
//!    among the subscriptions the user has.
//! 5. [`reasons`]: the identifiers of what fired, which SWAG and RRFlow read.
//!
//! The arithmetic and rules are configuration ([`config`]). Which models exist
//! and how they are used come from [`Knowledge`], which the host rebuilds as the
//! user works. Nothing here keeps prompt text, and scores are rule outputs rather
//! than calibrated probabilities.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::PreflightRuleConfig;

pub mod agents;
pub mod complexity;
pub mod config;
pub mod intent;
pub mod knowledge;
pub mod models;
pub mod reasons;
pub mod subscriptions;
#[cfg(test)]
mod tests;
pub mod text;
pub mod trained_level;

pub use agents::AgentResult;
pub use complexity::{ComplexityResult, FormulaTerm, LadderStep, bracket_for};
pub use config::{
    AssessmentConfig, Bracket, IntentWords, KnowledgeScanLimits, ModelMatching, PublishedModel,
    RoleRule, SignalWeights, StructuredInput, SubscriptionDefinition,
};
pub use intent::{IntentResult, WordHit};
pub use knowledge::{
    BracketUsage, Knowledge, KnowledgeSource, KnowledgeSummary, KnownModel, ObservedSubscription,
    SubscriptionCheck,
};
pub use models::{ModelResult, ModelSuggestion, known_models, with_roles};
pub use subscriptions::SubscriptionStatus;
pub use text::{contains_word, scored_text};
pub use trained_level::{ModelOutput, ModelScore, ModelScoreBinding};

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
    /// Present only when the configuration binds a trained model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_score: Option<ModelScore>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFacts {
    pub characters: usize,
    pub words: usize,
    pub tasks: Vec<String>,
    pub scoring_truncated: bool,
}

/// What the rules find in the text, before any trained model is consulted.
struct Scored {
    characters: usize,
    words: usize,
    tasks: Vec<String>,
    intents: Vec<intent::WordHit>,
    risks: Vec<intent::WordHit>,
    terms: Vec<FormulaTerm>,
    score: f64,
    bracket: usize,
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0 + 0.0
}

fn round2(value: f64) -> f64 {
    // Adding zero turns the -0.0 of an empty float sum into 0.0.
    (value * 100.0).round() / 100.0 + 0.0
}

fn score_prompt(config: &AssessmentConfig, rules: &PreflightRuleConfig, text: &str) -> Scored {
    let normalized = text::scored_slice(text, config.max_scored_bytes).to_lowercase();
    let words = text.split_whitespace().count();
    let intents = intent::intent_hits(config, rules, &normalized);
    let risks = intent::risk_hits(config, rules, &normalized);
    let tasks = text::task_clauses(&normalized, config);
    let (terms, score, bracket) =
        complexity::formula(config, &normalized, words, &intents, &tasks, &risks);
    Scored {
        characters: text.chars().count(),
        words,
        tasks,
        intents,
        risks,
        terms,
        score,
        bracket,
    }
}

pub fn assess(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &AssessRequest,
    knowledge: &Knowledge,
    checks: &BTreeMap<String, SubscriptionCheck>,
) -> Assessment {
    assess_with_model(config, rules, request, knowledge, checks, None)
}

/// [`assess`], with the trained model's output when the configuration binds one.
/// The model then chooses the bracket, and so the agents, role and effort; the
/// rule score is still reported in rule points. Without an output the rules
/// choose, and `modelScore.status` says the model was unavailable.
pub fn assess_with_model(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &AssessRequest,
    knowledge: &Knowledge,
    checks: &BTreeMap<String, SubscriptionCheck>,
    model: Option<&ModelOutput>,
) -> Assessment {
    let started = Instant::now();
    let text = scored_text(config, &request.text);
    let structured = matches!(text, std::borrow::Cow::Owned(_));
    let scored = score_prompt(config, rules, &text);

    let (chosen, model_score) = complexity::choose_bracket(config, scored.bracket, model);
    let agents = agents::agents_for(&config.brackets[chosen], scored.tasks.len());

    let subscriptions = subscriptions::subscription_statuses(config, request, knowledge, checks);
    let usable: BTreeSet<String> = subscriptions
        .iter()
        .filter(|s| s.usable)
        .map(|s| s.id.clone())
        .collect();
    let (models, models_known) = models::model_result(config, knowledge, &usable, chosen);

    let intent = intent::intent_result(rules, scored.intents, scored.risks);
    let complexity = complexity::complexity_result(config, scored.score, scored.terms, chosen);
    let fired = reasons::reasons(
        rules,
        &intent,
        &complexity,
        structured,
        model_score.as_ref(),
        &agents,
        &models,
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
            scoring_truncated: text.len() > config.max_scored_bytes,
        },
        intent,
        complexity,
        agents,
        models,
        subscriptions,
        knowledge: KnowledgeSummary {
            state: knowledge.state.clone(),
            refreshed_at: knowledge.refreshed_at.clone(),
            prompts_seen: knowledge.prompts_seen,
            models_known,
        },
        rules: fired,
        advisory: true,
        confidence: None,
        model_score,
    }
}
