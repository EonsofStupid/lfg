//! Reason identifiers: what fired, in a fixed order. They name rules, terms and
//! choices, never prompt words, so they are safe to retain and to send to SWAG.

use super::agents::AgentResult;
use super::complexity::ComplexityResult;
use super::intent::IntentResult;
use super::models::ModelResult;
use super::trained_level::ModelScore;
use crate::PreflightRuleConfig;

pub(super) fn reasons(
    rules: &PreflightRuleConfig,
    intent: &IntentResult,
    complexity: &ComplexityResult,
    structured: bool,
    model_score: Option<&ModelScore>,
    agents: &AgentResult,
    models: &ModelResult,
) -> Vec<String> {
    let mut fired: Vec<String> = intent
        .all
        .iter()
        .map(|hit| format!("intent:{}", hit.id))
        .collect();
    if intent.fallback {
        fired.push(format!("intent-fallback:{}", rules.fallback.intent));
    }
    fired.extend(intent.risks.iter().map(|hit| format!("risk:{}", hit.id)));
    fired.extend(
        complexity
            .terms
            .iter()
            .map(|term| format!("points:{}={}", term.id, term.points)),
    );
    if structured {
        fired.push("input:structured".into());
    }
    if let Some(score) = model_score {
        fired.push(format!("score-model:{}", score.status));
        if let Some(level) = score.level {
            fired.push(format!("score-model-level:{level}"));
            fired.push("bracket-source:model".into());
        }
    }
    fired.push(format!("bracket:{}", complexity.bracket));
    fired.push(format!("agents:{}", agents.total));
    fired.push(format!("role:{}", models.role));
    if let Some(model) = &models.suggested {
        fired.push(format!("model:{}/{}", model.subscription_id, model.model));
        fired.push(format!("model-origin:{}", model.origin));
        if let Some(effort) = &model.effort {
            fired.push(format!("effort:{effort}"));
        }
    }
    fired
}
