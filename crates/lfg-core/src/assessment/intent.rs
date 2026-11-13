//! Intent: what kind of work the prompt asks for, from the shared preflight
//! rules plus the assessment's extra words, and any sensitive content it touches.

use serde::{Deserialize, Serialize};

use super::config::AssessmentConfig;
use super::text::matched_words;
use crate::PreflightRuleConfig;

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

/// Every intent whose words appear in the lowercased prompt, in rule order.
pub(super) fn intent_hits(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    normalized: &str,
) -> Vec<WordHit> {
    let suffixes = &config.word_suffixes;
    rules
        .rules
        .iter()
        .filter_map(|rule| {
            let mut matched = matched_words(normalized, &rule.keywords, suffixes);
            for extra in config
                .extra_intent_words
                .iter()
                .filter(|extra| extra.intent == rule.intent)
            {
                matched.extend(matched_words(normalized, &extra.words, suffixes));
            }
            (!matched.is_empty()).then(|| WordHit {
                id: rule.intent.clone(),
                plain_label: rule.plain_label.clone(),
                matched,
            })
        })
        .collect()
}

/// Sensitive content the prompt touches, such as credentials or production data.
pub(super) fn risk_hits(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    normalized: &str,
) -> Vec<WordHit> {
    rules
        .risk_rules
        .iter()
        .filter_map(|rule| {
            let matched = matched_words(normalized, &rule.keywords, &config.word_suffixes);
            (!matched.is_empty()).then(|| WordHit {
                id: rule.risk.clone(),
                plain_label: rule.plain_label.clone(),
                matched,
            })
        })
        .collect()
}

/// The first matching intent is primary; with none, the configured fallback.
pub(super) fn intent_result(
    rules: &PreflightRuleConfig,
    all: Vec<WordHit>,
    risks: Vec<WordHit>,
) -> IntentResult {
    let fallback = all.is_empty();
    let (primary, plain_label) = all.first().map_or_else(
        || {
            (
                rules.fallback.intent.clone(),
                rules.fallback.plain_label.clone(),
            )
        },
        |hit| (hit.id.clone(), hit.plain_label.clone()),
    );
    IntentResult {
        primary,
        plain_label,
        all,
        risks,
        fallback,
    }
}
