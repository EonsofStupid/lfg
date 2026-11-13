//! The assessment configuration: words, weights, brackets, roles and limits.
//! Every number the classifier uses comes from here, and its digest is what
//! RRFlow and SWAG pin.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::trained_level::ModelScoreBinding;
use crate::digest_value;

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
    /// When the text is a JSON object with these fields, such as the task an
    /// RRFlow engine sends, assess only their values. Absent means the text is
    /// assessed as given. Present only when configured, so older digests hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_input: Option<StructuredInput>,
    /// A trained model whose work level chooses the bracket. Its manifest digest
    /// is part of this configuration, so pinning this digest also pins the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_score: Option<ModelScoreBinding>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StructuredInput {
    /// Fields read in this order; values that are not non-empty strings are skipped.
    pub json_fields: Vec<String>,
    pub separator: String,
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
        Self::from_json_str(include_str!("../../config/assessment.v1.json"))
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
        if let Some(structured) = &self.structured_input {
            let mut seen = BTreeSet::new();
            if structured.json_fields.is_empty()
                || structured
                    .json_fields
                    .iter()
                    .any(|field| field.trim().is_empty() || !seen.insert(field))
            {
                return Err("invalid structured input fields".into());
            }
        }
        if let Some(binding) = &self.model_score {
            binding.validate(&self.brackets)?;
        }
        Ok(())
    }
}
