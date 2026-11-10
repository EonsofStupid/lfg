//! Versioned, advisory-only assessment projection for integration clients.
//!
//! This deliberately does not consult local provider history, credentials or
//! subscription checks. Unknown request fields are accepted as opaque extension
//! data but are not used as authority, carried forward, or interpreted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::assessment::{
    AssessRequest, AssessmentConfig, Knowledge, ModelOutput, ModelScore, assess_with_model,
};
use crate::{PreflightRuleConfig, digest_text, digest_value};

pub const REQUEST_VERSION: &str = "lfg.observation-request.v1";
pub const RESPONSE_VERSION: &str = "lfg.observation.v1";

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationRequest {
    pub schema_version: String,
    pub request_id: String,
    pub revision: u64,
    pub text: String,
    pub input_sha256: String,
    pub expected_assessment_sha256: String,
    pub expected_rules_sha256: String,
    pub deadline_unix_ms: Option<u64>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    pub request_schema_version: &'static str,
    pub response_schema_version: &'static str,
    pub analyzer: &'static str,
    pub assessment_revision: String,
    pub assessment_sha256: String,
    pub rules_revision: String,
    pub rules_sha256: String,
    pub max_input_bytes: usize,
    pub max_scored_bytes: usize,
    pub max_response_bytes: usize,
    /// The trained model the configuration binds, when it binds one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<CapabilityModel>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityModel {
    pub id: String,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Observation {
    pub schema_version: &'static str,
    pub request_id: String,
    pub revision: u64,
    pub input_sha256: String,
    pub analyzer: &'static str,
    pub assessment_revision: String,
    pub assessment_sha256: String,
    pub rules_revision: String,
    pub rules_sha256: String,
    pub status: &'static str,
    pub intent: Intent,
    pub complexity: Complexity,
    pub agents: Agents,
    pub suggested_role: String,
    pub suggested_effort: String,
    pub reason_ids: Vec<String>,
    pub ignored_extension_count: usize,
    pub advisory: bool,
    pub confidence: Option<f64>,
    /// Present only when the configuration binds a trained model: what it
    /// chose and what the rules alone would have chosen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_score: Option<ModelScore>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Intent {
    pub status: &'static str,
    pub category: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Complexity {
    pub bracket: String,
    pub score: f64,
    pub score_unit: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Agents {
    pub suggested: usize,
    pub min: usize,
    pub max: usize,
}

pub fn capabilities(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    max_input_bytes: usize,
    max_response_bytes: usize,
) -> Capabilities {
    Capabilities {
        request_schema_version: REQUEST_VERSION,
        response_schema_version: RESPONSE_VERSION,
        analyzer: analyzer(config),
        assessment_revision: config.revision.clone(),
        assessment_sha256: config.digest(),
        rules_revision: rules.revision.clone(),
        rules_sha256: digest_value(&serde_json::to_value(rules).expect("rules serialize")),
        max_input_bytes,
        max_scored_bytes: config.max_scored_bytes,
        max_response_bytes,
        model: config.model_score.as_ref().map(|binding| CapabilityModel {
            id: binding.id.clone(),
            manifest_sha256: binding.manifest_sha256.clone(),
        }),
    }
}

fn analyzer(config: &AssessmentConfig) -> &'static str {
    if config.model_score.is_some() {
        "rules-and-trained-model"
    } else {
        "deterministic-rules"
    }
}

/// All failures are static codes so private input cannot enter error messages.
pub fn observe(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &ObservationRequest,
    max_input_bytes: usize,
    max_response_bytes: usize,
    now_unix_ms: u64,
) -> Result<Observation, &'static str> {
    observe_with_model(
        config,
        rules,
        request,
        max_input_bytes,
        max_response_bytes,
        now_unix_ms,
        None,
    )
}

/// The checks [`observe`] makes before assessing, so a host can refuse a
/// request before spending a model run on it.
pub fn check_request(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &ObservationRequest,
    max_input_bytes: usize,
    max_response_bytes: usize,
    now_unix_ms: u64,
) -> Result<(), &'static str> {
    if request.schema_version != REQUEST_VERSION {
        return Err("unsupported_schema");
    }
    if request.request_id.is_empty()
        || request.request_id.len() > max_input_bytes
        || !request
            .request_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        || request.text.trim().is_empty()
        || request.text.len() > max_input_bytes
    {
        return Err("invalid_input");
    }
    if request
        .deadline_unix_ms
        .is_some_and(|deadline| now_unix_ms >= deadline)
    {
        return Err("deadline_expired");
    }
    if request.input_sha256 != digest_text(&request.text) {
        return Err("input_digest_mismatch");
    }
    let caps = capabilities(config, rules, max_input_bytes, max_response_bytes);
    if request.expected_assessment_sha256 != caps.assessment_sha256
        || request.expected_rules_sha256 != caps.rules_sha256
    {
        return Err("configuration_changed");
    }
    Ok(())
}

/// [`observe`] with the trained model's output. A configuration that binds a
/// model never answers from the rules alone: without a valid output it fails.
pub fn observe_with_model(
    config: &AssessmentConfig,
    rules: &PreflightRuleConfig,
    request: &ObservationRequest,
    max_input_bytes: usize,
    max_response_bytes: usize,
    now_unix_ms: u64,
    model: Option<&ModelOutput>,
) -> Result<Observation, &'static str> {
    check_request(
        config,
        rules,
        request,
        max_input_bytes,
        max_response_bytes,
        now_unix_ms,
    )?;
    if config.model_score.is_some() && model.is_none() {
        return Err("model_unavailable");
    }
    let caps = capabilities(config, rules, max_input_bytes, max_response_bytes);
    let assessment = assess_with_model(
        config,
        rules,
        &AssessRequest {
            session_id: request.request_id.clone(),
            revision: request.revision,
            text: request.text.clone(),
            declared_subscriptions: Vec::new(),
        },
        &Knowledge::disabled(),
        &BTreeMap::new(),
        model,
    );
    if !assessment.complexity.score.is_finite() {
        return Err("nonfinite_score");
    }
    if assessment
        .model_score
        .as_ref()
        .is_some_and(|score| score.status != "available")
    {
        return Err("model_output_invalid");
    }
    let model_truncated = model.is_some_and(|output| output.truncated);
    let result = Observation {
        schema_version: RESPONSE_VERSION,
        request_id: request.request_id.clone(),
        revision: request.revision,
        input_sha256: request.input_sha256.clone(),
        analyzer: caps.analyzer,
        assessment_revision: caps.assessment_revision,
        assessment_sha256: caps.assessment_sha256,
        rules_revision: caps.rules_revision,
        rules_sha256: caps.rules_sha256,
        status: if assessment.prompt.scoring_truncated || model_truncated {
            "partial"
        } else {
            "available"
        },
        intent: Intent {
            status: if assessment.intent.fallback {
                "inconclusive"
            } else {
                "available"
            },
            category: assessment.intent.primary,
        },
        complexity: Complexity {
            bracket: assessment.complexity.bracket,
            score: assessment.complexity.score,
            score_unit: "configured-rule-points",
        },
        agents: Agents {
            suggested: assessment.agents.total,
            min: assessment.agents.min,
            max: assessment.agents.max,
        },
        suggested_role: assessment.models.role,
        suggested_effort: assessment.models.effort,
        reason_ids: assessment.rules,
        ignored_extension_count: request.extensions.len(),
        advisory: true,
        confidence: None,
        model_score: assessment.model_score,
    };
    let bytes = serde_json::to_vec(&result).map_err(|_| "serialization_failed")?;
    if bytes.len() > max_response_bytes {
        return Err("response_limit_exceeded");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::ModelScoreBinding;

    fn rules() -> PreflightRuleConfig {
        PreflightRuleConfig::from_json_str(include_str!("../config/preflight.v1.json")).unwrap()
    }

    fn bound() -> AssessmentConfig {
        let mut config = AssessmentConfig::default_alpha();
        config.model_score = Some(ModelScoreBinding {
            id: "lfg-dev-text-test".into(),
            manifest_sha256: "b".repeat(64),
            level_brackets: ["trivial", "simple", "moderate", "complex", "major"]
                .map(String::from)
                .to_vec(),
            level_cut_points: vec![0.5, 1.5, 2.5, 3.5],
        });
        config
    }

    fn request(config: &AssessmentConfig, text: &str) -> ObservationRequest {
        let caps = capabilities(config, &rules(), 12_000, 65_536);
        ObservationRequest {
            schema_version: REQUEST_VERSION.into(),
            request_id: "attempt-1".into(),
            revision: 3,
            text: text.into(),
            input_sha256: digest_text(text),
            expected_assessment_sha256: caps.assessment_sha256,
            expected_rules_sha256: caps.rules_sha256,
            deadline_unix_ms: None,
            extensions: BTreeMap::new(),
        }
    }

    fn output(probabilities: &[f64], truncated: bool) -> ModelOutput {
        ModelOutput {
            probabilities: probabilities.to_vec(),
            elapsed_micros: 900,
            truncated,
        }
    }

    #[test]
    fn unbound_configuration_keeps_the_rules_analyzer_and_shape() {
        let config = AssessmentConfig::default_alpha();
        let caps = capabilities(&config, &rules(), 12_000, 65_536);
        assert_eq!(caps.analyzer, "deterministic-rules");
        assert!(serde_json::to_value(&caps).unwrap().get("model").is_none());
        let observed = observe(
            &config,
            &rules(),
            &request(&config, "fix typo"),
            12_000,
            65_536,
            0,
        )
        .unwrap();
        let value = serde_json::to_value(&observed).unwrap();
        assert!(value.get("modelScore").is_none());
        assert_eq!(value["analyzer"], "deterministic-rules");
    }

    #[test]
    fn bound_configuration_needs_a_valid_model_output() {
        let config = bound();
        let req = request(&config, "fix typo");
        assert_eq!(
            observe(&config, &rules(), &req, 12_000, 65_536, 0).unwrap_err(),
            "model_unavailable"
        );
        assert_eq!(
            observe_with_model(
                &config,
                &rules(),
                &req,
                12_000,
                65_536,
                0,
                Some(&output(&[0.5, 0.4], false))
            )
            .unwrap_err(),
            "model_output_invalid"
        );
        let observed = observe_with_model(
            &config,
            &rules(),
            &req,
            12_000,
            65_536,
            0,
            Some(&output(&[0.0, 0.0, 0.0, 0.1, 0.9], false)),
        )
        .unwrap();
        assert_eq!(observed.analyzer, "rules-and-trained-model");
        assert_eq!(observed.status, "available");
        assert_eq!(observed.complexity.bracket, "major");
        assert_eq!(observed.complexity.score_unit, "configured-rule-points");
        assert!(observed.confidence.is_none() && observed.advisory);
        let major = config.brackets.iter().find(|b| b.id == "major").unwrap();
        assert_eq!(observed.suggested_role, major.role);
        assert_eq!(observed.suggested_effort, major.effort);
        assert!(
            observed.agents.min <= observed.agents.suggested
                && observed.agents.suggested <= observed.agents.max
        );
        let score = observed.model_score.as_ref().unwrap();
        assert_eq!(score.level, Some(4));
        let caps = capabilities(&config, &rules(), 12_000, 65_536);
        assert_eq!(caps.model.as_ref().unwrap().manifest_sha256, "b".repeat(64));
        let truncated = observe_with_model(
            &config,
            &rules(),
            &req,
            12_000,
            65_536,
            0,
            Some(&output(&[1.0, 0.0, 0.0, 0.0, 0.0], true)),
        )
        .unwrap();
        assert_eq!(truncated.status, "partial");
    }

    #[test]
    fn checks_run_before_any_model_and_pin_the_model() {
        let config = bound();
        let unbound = AssessmentConfig::default_alpha();
        // A request pinned to the rules-only digest is refused by a bound server.
        let stale = request(&unbound, "fix typo");
        assert_eq!(
            check_request(&config, &rules(), &stale, 12_000, 65_536, 0).unwrap_err(),
            "configuration_changed"
        );
        let mut other_model = config.clone();
        other_model.model_score.as_mut().unwrap().manifest_sha256 = "c".repeat(64);
        assert_eq!(
            check_request(
                &other_model,
                &rules(),
                &request(&config, "fix typo"),
                12_000,
                65_536,
                0
            )
            .unwrap_err(),
            "configuration_changed"
        );
        assert!(
            check_request(
                &config,
                &rules(),
                &request(&config, "fix typo"),
                12_000,
                65_536,
                0
            )
            .is_ok()
        );
    }
}
