//! Versioned, advisory-only assessment projection for integration clients.
//!
//! This deliberately does not consult local provider history, credentials or
//! subscription checks. Unknown request fields are accepted as opaque extension
//! data but are not used as authority, carried forward, or interpreted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::assessment::{AssessRequest, AssessmentConfig, Knowledge, assess};
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
        analyzer: "deterministic-rules",
        assessment_revision: config.revision.clone(),
        assessment_sha256: config.digest(),
        rules_revision: rules.revision.clone(),
        rules_sha256: digest_value(&serde_json::to_value(rules).expect("rules serialize")),
        max_input_bytes,
        max_scored_bytes: config.max_scored_bytes,
        max_response_bytes,
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
    let assessment = assess(
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
    );
    if !assessment.complexity.score.is_finite() {
        return Err("nonfinite_score");
    }
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
        status: if assessment.prompt.scoring_truncated {
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
    };
    let bytes = serde_json::to_vec(&result).map_err(|_| "serialization_failed")?;
    if bytes.len() > max_response_bytes {
        return Err("response_limit_exceeded");
    }
    Ok(result)
}
