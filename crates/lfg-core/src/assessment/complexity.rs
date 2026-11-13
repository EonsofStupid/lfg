//! Complexity: a configured points formula over the prompt, placed on a bracket
//! ladder. When a trained work-level model is bound and its output is valid, its
//! level chooses the bracket, and the rule bracket is reported beside it.

use serde::{Deserialize, Serialize};

use super::config::AssessmentConfig;
use super::intent::WordHit;
use super::text::matched_words;
use super::trained_level::{ModelOutput, ModelScore};
use super::{round2, round4, score_prompt};
use crate::PreflightRuleConfig;

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

/// The points formula: kinds of work, separate tasks, length, wide-reaching
/// words and sensitive content, each a value times its configured weight.
/// Returns the terms rounded for display, the unrounded score and the index of
/// the highest bracket whose minimum the score reaches.
pub(super) fn formula(
    config: &AssessmentConfig,
    normalized: &str,
    words: usize,
    intents: &[WordHit],
    tasks: &[String],
    risks: &[WordHit],
) -> (Vec<FormulaTerm>, f64, usize) {
    let scope = matched_words(normalized, &config.scope_words, &config.word_suffixes);
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
    let terms = terms
        .into_iter()
        .map(|term| FormulaTerm {
            points: round2(term.points),
            ..term
        })
        .collect();
    (terms, score, bracket)
}

/// The bracket the assessment uses and what the trained model contributed.
/// Without a bound model the rules choose and there is no model score; with
/// one, a valid output chooses, and otherwise the rules choose and the score
/// says the model was unavailable or its output invalid.
pub(super) fn choose_bracket(
    config: &AssessmentConfig,
    rules_index: usize,
    model: Option<&ModelOutput>,
) -> (usize, Option<ModelScore>) {
    let rules_bracket = config.brackets[rules_index].id.clone();
    match (&config.model_score, model) {
        (None, _) => (rules_index, None),
        (Some(binding), output) => {
            let base = ModelScore {
                status: "unavailable".into(),
                id: binding.id.clone(),
                manifest_sha256: binding.manifest_sha256.clone(),
                expected_level: None,
                probabilities: Vec::new(),
                level: None,
                bracket: None,
                rules_bracket,
                elapsed_micros: output.map(|o| o.elapsed_micros),
                truncated: output.map(|o| o.truncated),
            };
            match output.map(|o| (o, binding.choose(&config.brackets, &o.probabilities))) {
                Some((output, Some((expected, level, index)))) => (
                    index,
                    Some(ModelScore {
                        status: "available".into(),
                        expected_level: Some(round2(expected)),
                        probabilities: output.probabilities.iter().map(|p| round4(*p)).collect(),
                        level: Some(level),
                        bracket: Some(config.brackets[index].id.clone()),
                        ..base
                    }),
                ),
                Some((_, None)) => (
                    rules_index,
                    Some(ModelScore {
                        status: "invalid_output".into(),
                        ..base
                    }),
                ),
                None => (rules_index, Some(base)),
            }
        }
    }
}

/// The reported complexity: the rule score in rule points, the chosen bracket
/// and the whole ladder with the chosen step marked.
pub(super) fn complexity_result(
    config: &AssessmentConfig,
    score: f64,
    terms: Vec<FormulaTerm>,
    chosen: usize,
) -> ComplexityResult {
    let bracket = &config.brackets[chosen];
    ComplexityResult {
        score: round2(score),
        bracket: bracket.id.clone(),
        label: bracket.label.clone(),
        plain_label: bracket.plain_label.clone(),
        terms,
        ladder: config
            .brackets
            .iter()
            .enumerate()
            .map(|(index, step)| LadderStep {
                id: step.id.clone(),
                label: step.label.clone(),
                min_score: step.min_score,
                current: index == chosen,
            })
            .collect(),
    }
}

/// The rule bracket for a prompt, using the same formula as [`assess`](super::assess).
pub fn bracket_for(config: &AssessmentConfig, rules: &PreflightRuleConfig, text: &str) -> usize {
    score_prompt(config, rules, text).bracket
}
