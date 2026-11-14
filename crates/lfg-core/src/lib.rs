use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};
use uuid::Uuid;

pub mod assessment;
pub mod kernel;
pub mod observation;
pub mod operator_mood;

pub const PROTOCOL_VERSION: &str = "lfg.payload-signal.v0.3.0";
pub const CLOUD_EVENTS_VERSION: &str = "1.0";
pub const CONTROL_DATA_SCHEMA: &str = "urn:clyffy:schema:lfg:control-data:v1";
pub const SIGNAL_SOURCE: &str = "urn:clyffy:lfg:master-clyffy";
pub const MASTER_WORKSPACE_ID: &str = "clyffy.master";
pub const MASTER_PROJECT_KEY: &str = "shippin-master-clyffy";

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProtocolError {
    #[error("required protocol identity is missing or invalid")]
    InvalidIdentity,
    #[error("signal payload contains forbidden content field: {0}")]
    ForbiddenContent(String),
    #[error("signal payload hash does not match")]
    PayloadHashMismatch,
    #[error("signal event hash does not match")]
    EventHashMismatch,
    #[error("signal stream sequence or hash-chain continuation is invalid")]
    InvalidContinuation,
    #[error("plan graph is invalid: {0}")]
    InvalidPlan(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignalEnvelope {
    pub specversion: String,
    pub id: String,
    pub source: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub subject: String,
    pub time: String,
    pub datacontenttype: String,
    pub dataschema: String,
    pub traceparent: String,
    pub data: SignalData,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignalData {
    pub schema_version: String,
    pub sequence: u64,
    pub workspace_id: String,
    pub project_key: String,
    pub workflow_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding_id: Option<String>,
    pub correlation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causation_id: Option<String>,
    pub payload_class: String,
    pub content_policy: String,
    pub payload_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_hash: Option<String>,
    pub event_hash: String,
    pub payload: Value,
}

#[derive(Clone, Debug)]
pub struct SignalDraft {
    pub event_type: String,
    pub workflow_id: String,
    pub plan_id: Option<String>,
    pub task_id: Option<String>,
    pub agent_id: Option<String>,
    pub attempt_id: Option<String>,
    pub binding_id: Option<String>,
    pub causation_id: Option<String>,
    pub payload_class: String,
    pub payload: Value,
}

impl SignalEnvelope {
    pub fn seal(sequence: u64, previous_hash: Option<String>, draft: SignalDraft) -> Self {
        let id = format!("event-{}", Uuid::new_v4());
        let trace_id = digest_text(&draft.workflow_id)[..32].to_owned();
        let span_id = digest_text(&id)[..16].to_owned();
        let payload_hash = digest_value(&draft.payload);
        let mut signal = Self {
            specversion: CLOUD_EVENTS_VERSION.to_owned(),
            id: id.clone(),
            source: SIGNAL_SOURCE.to_owned(),
            event_type: draft.event_type,
            subject: format!("workflow/{}", draft.workflow_id),
            time: now_rfc3339(),
            datacontenttype: "application/json".to_owned(),
            dataschema: CONTROL_DATA_SCHEMA.to_owned(),
            traceparent: format!("00-{trace_id}-{span_id}-01"),
            data: SignalData {
                schema_version: PROTOCOL_VERSION.to_owned(),
                sequence,
                workspace_id: MASTER_WORKSPACE_ID.to_owned(),
                project_key: MASTER_PROJECT_KEY.to_owned(),
                workflow_id: draft.workflow_id.clone(),
                plan_id: draft.plan_id,
                task_id: draft.task_id,
                agent_id: draft.agent_id,
                attempt_id: draft.attempt_id,
                binding_id: draft.binding_id,
                correlation_id: draft.workflow_id,
                causation_id: draft.causation_id,
                payload_class: draft.payload_class,
                content_policy: "metadata-only".to_owned(),
                payload_hash,
                previous_hash,
                event_hash: String::new(),
                payload: draft.payload,
            },
        };
        signal.data.event_hash = signal.computed_event_hash();
        signal
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.specversion != CLOUD_EVENTS_VERSION
            || self.data.schema_version != PROTOCOL_VERSION
            || self.source.is_empty()
            || self.id.is_empty()
            || self.event_type.is_empty()
            || self.data.workflow_id.is_empty()
            || self.data.workspace_id.is_empty()
            || self.traceparent.len() != 55
        {
            return Err(ProtocolError::InvalidIdentity);
        }
        reject_forbidden_content(&self.data.payload, "payload")?;
        if self.data.payload_hash != digest_value(&self.data.payload) {
            return Err(ProtocolError::PayloadHashMismatch);
        }
        if self.data.event_hash != self.computed_event_hash() {
            return Err(ProtocolError::EventHashMismatch);
        }
        Ok(())
    }

    pub fn computed_event_hash(&self) -> String {
        let mut clone = self.clone();
        clone.data.event_hash.clear();
        digest_value(&serde_json::to_value(clone).expect("signal serialization is infallible"))
    }
}

/// Checks that `signal` correctly continues from `tail` — the `(sequence,
/// event_hash)` of the most recent prior signal on the same workflow, or
/// `None` if this is the workflow's first signal. Callers that already
/// track a running tail per workflow (e.g. an append-only store) can use
/// this directly instead of re-deriving it from the full history via
/// [`validate_signal_stream`], which is O(n) per call.
pub fn check_continuation(
    tail: Option<(u64, &str)>,
    signal: &SignalEnvelope,
) -> Result<(), ProtocolError> {
    match tail {
        Some((sequence, hash)) => {
            if signal.data.sequence != sequence + 1
                || signal.data.previous_hash.as_deref() != Some(hash)
            {
                return Err(ProtocolError::InvalidContinuation);
            }
        }
        None => {
            if signal.data.sequence != 0 || signal.data.previous_hash.is_some() {
                return Err(ProtocolError::InvalidContinuation);
            }
        }
    }
    Ok(())
}

pub fn validate_signal_stream(signals: &[SignalEnvelope]) -> Result<(), ProtocolError> {
    let mut tails: BTreeMap<&str, (u64, &str)> = BTreeMap::new();
    for signal in signals {
        signal.validate()?;
        check_continuation(tails.get(signal.data.workflow_id.as_str()).copied(), signal)?;
        tails.insert(
            signal.data.workflow_id.as_str(),
            (signal.data.sequence, signal.data.event_hash.as_str()),
        );
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestRef {
    pub id: String,
    pub version: String,
    pub sha256: String,
}

impl ManifestRef {
    pub fn new(id: impl Into<String>, version: impl Into<String>, definition: &str) -> Self {
        let id = id.into();
        let version = version.into();
        let sha256 = digest_text(&format!("{id}\u{1f}{version}\u{1f}{definition}"));
        Self {
            id,
            version,
            sha256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RroCategoryManifest {
    pub id: String,
    pub version: String,
    pub label: String,
    pub plain_label: String,
    pub capability: String,
    pub tone: String,
    pub hash: String,
}

impl RroCategoryManifest {
    pub fn new(id: &str, label: &str, plain_label: &str, capability: &str, tone: &str) -> Self {
        let mut value = Self {
            id: id.to_owned(),
            version: "1.0.0".to_owned(),
            label: label.to_owned(),
            plain_label: plain_label.to_owned(),
            capability: capability.to_owned(),
            tone: tone.to_owned(),
            hash: String::new(),
        };
        value.hash = digest_value(&serde_json::to_value(&value).expect("manifest serializes"));
        value
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasonReadyObject {
    pub id: String,
    pub category: RroCategoryManifest,
    pub confidence: Option<f64>,
    pub advisory: bool,
    pub lineage: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentSpec {
    pub id: String,
    pub role: ManifestRef,
    pub required_capabilities: Vec<String>,
    pub max_parallel_tasks: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSpec {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub rro_ids: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub side_effect_class: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEdge {
    pub from_task_id: String,
    pub to_task_id: String,
    pub condition: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamPlan {
    pub schema_version: String,
    pub id: String,
    pub revision: u32,
    pub workspace_id: String,
    pub project_key: String,
    pub compiler: ManifestRef,
    pub policy: ManifestRef,
    pub rros: Vec<ReasonReadyObject>,
    pub agents: Vec<AgentSpec>,
    pub tasks: Vec<TaskSpec>,
    pub edges: Vec<TaskEdge>,
    pub max_parallel_tasks: u32,
    pub created_at: String,
    pub hash: String,
}

impl TeamPlan {
    pub fn seal(&mut self) {
        self.hash.clear();
        self.hash = digest_value(&serde_json::to_value(&self).expect("plan serializes"));
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.schema_version != "lfg.team-plan.v1"
            || self.id.is_empty()
            || self.tasks.is_empty()
            || self.agents.is_empty()
            || self.max_parallel_tasks == 0
        {
            return Err(ProtocolError::InvalidPlan(
                "identity or required collection".into(),
            ));
        }
        let agents: BTreeSet<_> = self.agents.iter().map(|agent| agent.id.as_str()).collect();
        let tasks: BTreeSet<_> = self.tasks.iter().map(|task| task.id.as_str()).collect();
        if agents.len() != self.agents.len() || tasks.len() != self.tasks.len() {
            return Err(ProtocolError::InvalidPlan("duplicate agent or task".into()));
        }
        if self
            .tasks
            .iter()
            .any(|task| !agents.contains(task.agent_id.as_str()))
            || self.edges.iter().any(|edge| {
                !tasks.contains(edge.from_task_id.as_str())
                    || !tasks.contains(edge.to_task_id.as_str())
            })
        {
            return Err(ProtocolError::InvalidPlan(
                "dangling graph reference".into(),
            ));
        }
        let mut clone = self.clone();
        clone.hash.clear();
        if self.hash != digest_value(&serde_json::to_value(clone).expect("plan serializes")) {
            return Err(ProtocolError::InvalidPlan("content hash mismatch".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderTarget {
    pub target_id: String,
    pub display_name: String,
    pub provider: String,
    pub adapter_id: String,
    pub model: String,
    pub provenance: String,
    pub availability: String,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderBinding {
    pub id: String,
    pub plan_id: String,
    pub task_id: String,
    pub agent_id: String,
    pub attempt_id: String,
    pub target: ProviderTarget,
    pub committed_at: String,
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationRubric {
    pub id: String,
    pub version: String,
    pub criteria: BTreeMap<String, f64>,
    pub pass_threshold: f64,
    #[serde(default)]
    pub hash: String,
}

impl EvaluationRubric {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let mut value: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let total: f64 = value.criteria.values().sum();
        if value.id.trim().is_empty()
            || value.version.trim().is_empty()
            || value.criteria.is_empty()
            || value.criteria.iter().any(|(name, weight)| {
                name.trim().is_empty() || !weight.is_finite() || *weight < 0.0
            })
            || (total - 1.0).abs() > 1e-9
            || !value.pass_threshold.is_finite()
            || !(0.0..=1.0).contains(&value.pass_threshold)
        {
            return Err("invalid evaluation rubric".into());
        }
        value.hash.clear();
        value.hash = digest_value(&serde_json::to_value(&value).expect("rubric serializes"));
        Ok(value)
    }

    pub fn default_alpha() -> Self {
        Self::from_json_str(include_str!("../config/evaluation-rubric.v1.json"))
            .expect("bundled evaluation rubric is valid")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionPolicy {
    pub id: String,
    pub version: String,
    pub minimum_samples_per_arm: usize,
    pub min_independent_graders: usize,
    pub minimum_score_lift: f64,
    pub minimum_pass_rate: f64,
    pub maximum_failure_rate: f64,
}

impl PromotionPolicy {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if value.id.trim().is_empty()
            || value.version.trim().is_empty()
            || value.minimum_samples_per_arm == 0
            || value.min_independent_graders == 0
            || [
                value.minimum_score_lift,
                value.minimum_pass_rate,
                value.maximum_failure_rate,
            ]
            .iter()
            .any(|n| !n.is_finite() || !(0.0..=1.0).contains(n))
        {
            return Err("invalid promotion policy".into());
        }
        Ok(value)
    }

    pub fn default_alpha() -> Self {
        Self::from_json_str(include_str!("../config/promotion-policy.v1.json"))
            .expect("bundled promotion policy is valid")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGradeRequest {
    pub experiment_id: String,
    pub variant_id: String,
    pub outcome_id: String,
    pub grader_run_id: String,
    pub scores: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGrade {
    pub schema_version: String,
    pub id: String,
    pub experiment_id: String,
    pub variant_id: String,
    pub outcome_id: String,
    pub grader_run_id: String,
    pub rubric: ManifestRef,
    pub scores: BTreeMap<String, f64>,
    pub overall_score: f64,
    pub verdict: String,
    pub recorded_at: String,
    pub hash: String,
}

pub fn grade_request(
    request: SemanticGradeRequest,
    rubric: &EvaluationRubric,
) -> Result<SemanticGrade, ProtocolError> {
    if request.experiment_id.trim().is_empty()
        || request.variant_id.trim().is_empty()
        || request.outcome_id.trim().is_empty()
        || request.grader_run_id.trim().is_empty()
        || request.scores.len() != rubric.criteria.len()
    {
        return Err(ProtocolError::InvalidPlan(
            "grade identity or score set is invalid".into(),
        ));
    }
    if request.grader_run_id == request.outcome_id {
        return Err(ProtocolError::InvalidPlan(
            "the evaluated outcome cannot grade itself".into(),
        ));
    }
    let mut overall = 0.0;
    for (criterion, weight) in &rubric.criteria {
        let Some(score) = request.scores.get(criterion) else {
            return Err(ProtocolError::InvalidPlan(format!(
                "missing grade criterion {criterion}"
            )));
        };
        if !(0.0..=1.0).contains(score) {
            return Err(ProtocolError::InvalidPlan(format!(
                "grade criterion {criterion} is outside 0..1"
            )));
        }
        overall += score * weight;
    }
    let mut grade = SemanticGrade {
        schema_version: "lfg.semantic-grade.v1".into(),
        id: format!("grade-{}", Uuid::new_v4()),
        experiment_id: request.experiment_id,
        variant_id: request.variant_id,
        outcome_id: request.outcome_id,
        grader_run_id: request.grader_run_id,
        rubric: ManifestRef {
            id: rubric.id.clone(),
            version: rubric.version.clone(),
            sha256: rubric.hash.clone(),
        },
        scores: request.scores,
        overall_score: overall,
        verdict: if overall >= rubric.pass_threshold {
            "pass".into()
        } else {
            "fail".into()
        },
        recorded_at: now_rfc3339(),
        hash: String::new(),
    };
    grade.hash = digest_value(&serde_json::to_value(&grade).expect("grade serializes"));
    Ok(grade)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariantStatistics {
    pub variant_id: String,
    pub samples: usize,
    /// Grader-run IDs are caller-supplied; distinct runs do not prove independence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grader_runs: Option<usize>,
    /// Number whose independence is verified. The current grade protocol cannot verify it.
    pub independent_graders: usize,
    pub mean_score: f64,
    pub pass_rate: f64,
    pub failure_rate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentStatistics {
    pub experiment_id: String,
    pub variants: Vec<VariantStatistics>,
    pub computed_at: String,
}

pub fn experiment_statistics(
    experiment_id: &str,
    grades: &[SemanticGrade],
) -> ExperimentStatistics {
    let mut grouped = BTreeMap::<String, Vec<&SemanticGrade>>::new();
    for grade in grades
        .iter()
        .filter(|grade| grade.experiment_id == experiment_id)
    {
        grouped
            .entry(grade.variant_id.clone())
            .or_default()
            .push(grade);
    }
    let variants = grouped
        .into_iter()
        .map(|(variant_id, values)| {
            let mut outcomes = BTreeMap::<&str, Vec<&SemanticGrade>>::new();
            for grade in &values {
                outcomes.entry(&grade.outcome_id).or_default().push(grade);
            }
            let graders = values
                .iter()
                .map(|grade| grade.grader_run_id.as_str())
                .collect::<BTreeSet<_>>();
            let samples = outcomes.len();
            let mean_score = if outcomes.is_empty() {
                0.0
            } else {
                outcomes
                    .values()
                    .map(|grades| {
                        grades.iter().map(|grade| grade.overall_score).sum::<f64>()
                            / grades.len() as f64
                    })
                    .sum::<f64>()
                    / samples as f64
            };
            let pass_rate = if outcomes.is_empty() {
                0.0
            } else {
                outcomes
                    .values()
                    .map(|grades| {
                        grades
                            .iter()
                            .filter(|grade| grade.verdict == "pass")
                            .count() as f64
                            / grades.len() as f64
                    })
                    .sum::<f64>()
                    / samples as f64
            };
            VariantStatistics {
                variant_id,
                samples,
                grader_runs: Some(graders.len()),
                // Run IDs do not identify graders or provide evidence of independence.
                independent_graders: 0,
                mean_score,
                pass_rate,
                failure_rate: 1.0 - pass_rate,
            }
        })
        .collect();
    ExperimentStatistics {
        experiment_id: experiment_id.into(),
        variants,
        computed_at: now_rfc3339(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionDecisionRequest {
    pub experiment_id: String,
    pub baseline_variant_id: String,
    pub candidate_variant_id: String,
    pub action: String,
    pub decided_by: ManifestRef,
    #[serde(default)]
    pub reason_codes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionDecision {
    pub schema_version: String,
    pub id: String,
    pub experiment_id: String,
    pub baseline_variant_id: String,
    pub candidate_variant_id: String,
    pub policy: PromotionPolicy,
    pub statistics: ExperimentStatistics,
    pub recommendation: String,
    pub action: String,
    pub decided_by: ManifestRef,
    pub reason_codes: Vec<String>,
    pub decided_at: String,
    pub hash: String,
}

pub fn promotion_recommendation(
    statistics: &ExperimentStatistics,
    baseline: &str,
    candidate: &str,
    policy: &PromotionPolicy,
) -> String {
    if baseline.trim().is_empty() || candidate.trim().is_empty() || baseline == candidate {
        return "invalid-comparison".into();
    }
    let baseline = statistics
        .variants
        .iter()
        .find(|variant| variant.variant_id == baseline);
    let candidate = statistics
        .variants
        .iter()
        .find(|variant| variant.variant_id == candidate);
    let (Some(baseline), Some(candidate)) = (baseline, candidate) else {
        return "insufficient-sample".into();
    };
    if [baseline, candidate]
        .iter()
        .any(|variant| variant.samples < policy.minimum_samples_per_arm)
    {
        return "insufficient-sample".into();
    }
    if [baseline, candidate]
        .iter()
        .any(|variant| variant.independent_graders < policy.min_independent_graders)
    {
        return "insufficient-independent-evidence".into();
    }
    if candidate.pass_rate < policy.minimum_pass_rate
        || candidate.failure_rate > policy.maximum_failure_rate
    {
        return "hold".into();
    }
    if candidate.mean_score - baseline.mean_score >= policy.minimum_score_lift {
        "promote".into()
    } else {
        "hold".into()
    }
}

impl ProviderBinding {
    pub fn seal(&mut self) {
        self.hash.clear();
        self.hash = digest_value(&serde_json::to_value(&self).expect("binding serializes"));
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightRequest {
    pub session_id: String,
    pub revision: u64,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightSnapshot {
    pub schema_version: String,
    pub session_id: String,
    pub revision: u64,
    pub mode: String,
    pub model: ManifestRef,
    pub duration_micros: u64,
    pub intents: Vec<Prediction>,
    pub capabilities: Vec<Prediction>,
    pub risks: Vec<Prediction>,
    pub rros: Vec<ReasonReadyObject>,
    pub predicted_agents: usize,
    pub predicted_branches: usize,
    pub confidence: Option<f64>,
    pub abstained: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Prediction {
    pub label: String,
    pub confidence: Option<f64>,
    pub source: String,
}

pub trait PreflightEncoder: Send + Sync {
    fn manifest(&self) -> ManifestRef;
    fn encode(&self, request: &PreflightRequest) -> PreflightSnapshot;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightRule {
    pub keywords: Vec<String>,
    pub intent: String,
    pub capability: String,
    pub label: String,
    pub plain_label: String,
    pub tone: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightFallback {
    pub intent: String,
    pub capability: String,
    pub label: String,
    pub plain_label: String,
    pub tone: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightRiskRule {
    pub keywords: Vec<String>,
    pub risk: String,
    pub category_id: String,
    pub capability: String,
    pub label: String,
    pub plain_label: String,
    pub tone: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightRuleConfig {
    pub schema_version: String,
    pub revision: String,
    pub max_branches: usize,
    pub min_agents: usize,
    pub rules: Vec<PreflightRule>,
    pub fallback: PreflightFallback,
    pub risk_rules: Vec<PreflightRiskRule>,
}

impl PreflightRuleConfig {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if config.schema_version != "lfg.preflight-rules.v1"
            || config.revision.trim().is_empty()
            || config.max_branches == 0
            || config.min_agents == 0
            || config.rules.is_empty()
        {
            return Err("invalid preflight schema, revision, limits, or rules".into());
        }
        let valid_words = |words: &[String]| {
            !words.is_empty()
                && words
                    .iter()
                    .all(|w| !w.trim().is_empty() && *w == w.to_lowercase())
        };
        let valid_fields = |fields: &[&str]| fields.iter().all(|value| !value.trim().is_empty());
        let mut ids = BTreeSet::new();
        for r in &config.rules {
            if !valid_words(&r.keywords)
                || !valid_fields(&[&r.intent, &r.capability, &r.label, &r.plain_label, &r.tone])
                || !ids.insert(&r.intent)
            {
                return Err("invalid or duplicate preflight rule".into());
            }
        }
        let f = &config.fallback;
        if !valid_fields(&[&f.intent, &f.capability, &f.label, &f.plain_label, &f.tone])
            || !ids.insert(&f.intent)
        {
            return Err("invalid or duplicate preflight fallback".into());
        }
        for r in &config.risk_rules {
            if !valid_words(&r.keywords)
                || !valid_fields(&[
                    &r.risk,
                    &r.category_id,
                    &r.capability,
                    &r.label,
                    &r.plain_label,
                    &r.tone,
                ])
                || !ids.insert(&r.category_id)
            {
                return Err("invalid or duplicate preflight risk rule".into());
            }
        }
        Ok(config)
    }
}

#[derive(Clone, Debug)]
pub struct CpuRuleEncoder {
    config: PreflightRuleConfig,
    manifest: ManifestRef,
}

impl CpuRuleEncoder {
    pub fn new(config: PreflightRuleConfig) -> Result<Self, String> {
        let encoded = serde_json::to_string(&config).map_err(|error| error.to_string())?;
        let config = PreflightRuleConfig::from_json_str(&encoded)?;
        let manifest = ManifestRef::new("lfg-cpu-rule-baseline", &config.revision, &encoded);
        Ok(Self { config, manifest })
    }

    pub fn config(&self) -> &PreflightRuleConfig {
        &self.config
    }
}

impl Default for CpuRuleEncoder {
    fn default() -> Self {
        let config =
            PreflightRuleConfig::from_json_str(include_str!("../config/preflight.v1.json"))
                .expect("bundled preflight configuration is valid");
        Self::new(config).expect("bundled preflight configuration is valid")
    }
}

impl PreflightEncoder for CpuRuleEncoder {
    fn manifest(&self) -> ManifestRef {
        self.manifest.clone()
    }

    fn encode(&self, request: &PreflightRequest) -> PreflightSnapshot {
        let started = Instant::now();
        let normalized = request.text.to_lowercase();
        let mut intents = Vec::new();
        let mut capabilities = BTreeSet::<String>::new();
        let mut risks = Vec::new();
        let mut categories = Vec::<RroCategoryManifest>::new();
        for rule in &self.config.rules {
            if rule.keywords.iter().any(|word| normalized.contains(word)) {
                intents.push(Prediction {
                    label: rule.intent.clone(),
                    confidence: None,
                    source: "cpu-rule".into(),
                });
                capabilities.insert(rule.capability.clone());
                categories.push(RroCategoryManifest::new(
                    &rule.intent,
                    &rule.label,
                    &rule.plain_label,
                    &rule.capability,
                    &rule.tone,
                ));
            }
        }
        if intents.is_empty() && !normalized.trim().is_empty() {
            let f = &self.config.fallback;
            intents.push(Prediction {
                label: f.intent.clone(),
                confidence: None,
                source: "cpu-rule".into(),
            });
            capabilities.insert(f.capability.clone());
            categories.push(RroCategoryManifest::new(
                &f.intent,
                &f.label,
                &f.plain_label,
                &f.capability,
                &f.tone,
            ));
        }
        for rule in &self.config.risk_rules {
            if rule.keywords.iter().any(|word| normalized.contains(word)) {
                risks.push(Prediction {
                    label: rule.risk.clone(),
                    confidence: None,
                    source: "cpu-rule".into(),
                });
                categories.push(RroCategoryManifest::new(
                    &rule.category_id,
                    &rule.label,
                    &rule.plain_label,
                    &rule.capability,
                    &rule.tone,
                ));
            }
        }
        categories.sort_by(|a, b| a.id.cmp(&b.id));
        categories.dedup_by(|a, b| a.id == b.id);
        let rros = categories
            .into_iter()
            .enumerate()
            .map(|(index, category)| ReasonReadyObject {
                id: format!("rro-preview-{}-{index}", request.revision),
                category,
                confidence: None,
                advisory: true,
                lineage: vec![format!(
                    "session/{}/revision/{}",
                    request.session_id, request.revision
                )],
            })
            .collect::<Vec<_>>();
        let predicted_agents = intents.len().max(self.config.min_agents);
        let predicted_branches = predicted_agents.min(self.config.max_branches);
        PreflightSnapshot {
            schema_version: "lfg.preflight.v2".into(),
            session_id: request.session_id.clone(),
            revision: request.revision,
            mode: "cpu-rule-baseline".into(),
            model: self.manifest(),
            duration_micros: started.elapsed().as_micros() as u64,
            intents,
            capabilities: capabilities
                .into_iter()
                .map(|label| Prediction {
                    label,
                    confidence: None,
                    source: "cpu-rule".into(),
                })
                .collect(),
            risks,
            rros,
            predicted_agents,
            predicted_branches,
            confidence: None,
            abstained: true,
        }
    }
}

/// A configured model profile. Its scores are operator-supplied relative
/// estimates, not calibrated probabilities or independently verified results.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EscalationTarget {
    pub provider: String,
    pub model: String,
    pub role: String,
    pub raw_capability: f64,
    pub tool_loop_efficiency: f64,
    pub cost_tier: String,
    pub subscription_id: String,
    pub strengths: Vec<String>,
    pub effort_levels: Vec<String>,
    pub evidence: ModelProfileEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProfileEvidence {
    pub source: String,
    pub basis: String,
    pub observed_at: String,
}

/// One catalog entry as returned to a caller: the target plus whether the
/// caller's currently-active subscriptions actually make it usable right
/// now. The full catalog is always returned so a UI can show what becomes
/// available once a subscription is activated, not just what's active today.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EscalationCandidate {
    pub target: EscalationTarget,
    pub active: bool,
}

/// Select among active subscriptions using the explicitly configured scoring policy.
pub fn select_escalation(
    catalog: &[EscalationTarget],
    active_subscriptions: &[String],
    routing_need: &str,
    scoring: &ClassifierScoring,
) -> Option<EscalationTarget> {
    let available = catalog.iter().filter(|target| {
        active_subscriptions
            .iter()
            .any(|id| id == &target.subscription_id)
            && scoring.cost_ranks.contains_key(&target.cost_tier)
    });
    match routing_need {
        "capability" => available.max_by(|a, b| a.raw_capability.total_cmp(&b.raw_capability)),
        "efficiency" => {
            available.max_by(|a, b| a.tool_loop_efficiency.total_cmp(&b.tool_loop_efficiency))
        }
        "cost" => available.min_by_key(|target| scoring.cost_ranks[&target.cost_tier]),
        "balanced" => available.max_by(|a, b| {
            let score = |target: &EscalationTarget| {
                target.raw_capability * scoring.capability_weight
                    + target.tool_loop_efficiency * scoring.efficiency_weight
                    - f64::from(scoring.cost_ranks[&target.cost_tier]) * scoring.cost_weight
            };
            score(a).total_cmp(&score(b))
        }),
        _ => None,
    }
    .cloned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryDefinition {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierConfig {
    pub schema_version: String,
    pub revision: String,
    pub project: String,
    pub default_routing_need: String,
    pub categories: Vec<CategoryDefinition>,
    pub escalation_category_id: String,
    pub escalation_percent_threshold: f64,
    pub escalation_token_threshold: usize,
    pub escalation_catalog: Vec<EscalationTarget>,
    pub scoring: ClassifierScoring,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifierScoring {
    pub keyword_weight: f64,
    pub length_divisor: f64,
    pub length_cap: f64,
    pub minimum_raw_score: f64,
    pub agents_per_words: usize,
    pub min_escalation_agents: usize,
    pub max_escalation_agents: usize,
    pub normal_agents: usize,
    pub capability_weight: f64,
    pub efficiency_weight: f64,
    pub cost_weight: f64,
    pub cost_ranks: BTreeMap<String, u8>,
}

impl ClassifierConfig {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "lfg.classifier-config.v2"
            || self.project.trim().is_empty()
            || self.revision.trim().is_empty()
            || !["balanced", "capability", "efficiency", "cost"]
                .contains(&self.default_routing_need.as_str())
        {
            return Err("unsupported classifier schema or empty project".into());
        }
        let mut ids = BTreeSet::new();
        for category in &self.categories {
            if category.id.trim().is_empty()
                || category.label.trim().is_empty()
                || !ids.insert(&category.id)
                || !category.weight.is_finite()
                || category.weight < 0.0
                || category
                    .keywords
                    .iter()
                    .any(|word| word.trim().is_empty() || *word != word.to_lowercase())
            {
                return Err("invalid category, duplicate ID, weight, or keyword".into());
            }
        }
        if !ids.contains(&self.escalation_category_id)
            || !self.escalation_percent_threshold.is_finite()
            || !(0.0..=100.0).contains(&self.escalation_percent_threshold)
            || self.escalation_token_threshold == 0
        {
            return Err("invalid escalation category or threshold".into());
        }
        let p = &self.scoring;
        let numbers = [
            p.keyword_weight,
            p.length_divisor,
            p.length_cap,
            p.minimum_raw_score,
            p.capability_weight,
            p.efficiency_weight,
            p.cost_weight,
        ];
        if numbers.iter().any(|x| !x.is_finite() || *x < 0.0)
            || p.length_divisor == 0.0
            || p.agents_per_words == 0
            || p.normal_agents == 0
            || p.min_escalation_agents == 0
            || p.max_escalation_agents < p.min_escalation_agents
            || p.cost_ranks.is_empty()
        {
            return Err("invalid scoring policy".into());
        }
        for target in &self.escalation_catalog {
            if target.provider.trim().is_empty()
                || target.model.trim().is_empty()
                || target.subscription_id.trim().is_empty()
                || !p.cost_ranks.contains_key(&target.cost_tier)
                || target.evidence.source.trim().is_empty()
                || target.evidence.basis.trim().is_empty()
                || target.evidence.observed_at.trim().is_empty()
                || target
                    .strengths
                    .iter()
                    .chain(&target.effort_levels)
                    .any(|value| value.trim().is_empty())
                || !target.raw_capability.is_finite()
                || !(0.0..=1.0).contains(&target.raw_capability)
                || !target.tool_loop_efficiency.is_finite()
                || !(0.0..=1.0).contains(&target.tool_loop_efficiency)
            {
                return Err("invalid escalation target".into());
            }
        }
        Ok(())
    }

    pub fn default_alpha() -> Self {
        Self::from_json_str(include_str!("../config/classifier.v2.json"))
            .expect("bundled classifier configuration is valid")
    }
}

/// Distinct from `PreflightRequest` (the older RRO/agent-count encoder's
/// input) because `active_subscriptions`/`routing_need` are specific to the
/// category classifier and shouldn't leak into that unrelated contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassifyRequest {
    pub session_id: String,
    pub revision: u64,
    pub text: String,
    /// Automaton provider ids the caller currently has an active seat for
    /// (e.g. `claude`, `codex`, `grok`, `cursor`). Empty means nothing is
    /// active yet — the full catalog still comes back, just with every
    /// candidate flagged inactive.
    #[serde(default)]
    pub active_subscriptions: Vec<String>,
    /// `capability` | `efficiency` | `cost` | `balanced` (default).
    #[serde(default)]
    pub routing_need: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CategoryScore {
    pub id: String,
    pub label: String,
    pub percent: f64,
    pub matched_keywords: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassificationSnapshot {
    pub schema_version: String,
    pub config_revision: String,
    pub config_digest: String,
    pub session_id: String,
    pub revision: u64,
    pub duration_micros: u64,
    pub token_estimate: usize,
    pub categories: Vec<CategoryScore>,
    pub needs_bigger_model: bool,
    pub predicted_subagents: usize,
    /// The full catalog, each flagged active/inactive against the caller's
    /// `active_subscriptions` — so a UI can show what's usable now and what
    /// becomes available once a subscription is activated.
    pub escalation_catalog: Vec<EscalationCandidate>,
    /// The best *active* candidate for the request's `routing_need`. `None`
    /// when nothing escalation-eligible has an active subscription yet, or
    /// escalation wasn't triggered at all.
    pub recommended_escalation: Option<EscalationTarget>,
    pub routing_need: String,
    pub project: String,
}

pub trait CategoryClassifier: Send + Sync {
    fn manifest(&self) -> ManifestRef;
    fn classify(
        &self,
        request: &ClassifyRequest,
        config: &ClassifierConfig,
    ) -> ClassificationSnapshot;
}

#[derive(Clone, Debug, Default)]
pub struct CpuCategoryClassifier;

impl CategoryClassifier for CpuCategoryClassifier {
    fn manifest(&self) -> ManifestRef {
        ManifestRef::new(
            "lfg-cpu-category-baseline",
            "1.0.0",
            "deterministic keyword-weighted category baseline; advisory, pending trained replacement",
        )
    }

    fn classify(
        &self,
        request: &ClassifyRequest,
        config: &ClassifierConfig,
    ) -> ClassificationSnapshot {
        let started = Instant::now();
        let normalized = request.text.to_lowercase();
        let token_estimate = request.text.split_whitespace().count();
        let routing_need = if request.routing_need.is_empty() {
            &config.default_routing_need
        } else {
            &request.routing_need
        };

        let scored = config
            .categories
            .iter()
            .map(|category| {
                let matched = category
                    .keywords
                    .iter()
                    .filter(|keyword| normalized.contains(keyword.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                let length_bonus = if category.id == config.escalation_category_id {
                    (token_estimate as f64 / config.scoring.length_divisor)
                        .min(config.scoring.length_cap)
                } else {
                    0.0
                };
                let raw = (category.weight
                    + matched.len() as f64 * config.scoring.keyword_weight
                    + length_bonus)
                    .max(config.scoring.minimum_raw_score);
                (category.id.clone(), category.label.clone(), raw, matched)
            })
            .collect::<Vec<_>>();
        let total: f64 = scored.iter().map(|(_, _, raw, _)| raw).sum();
        let categories = scored
            .into_iter()
            .map(|(id, label, raw, matched_keywords)| CategoryScore {
                id,
                label,
                percent: if total > 0.0 {
                    (raw / total * 10000.0).round() / 100.0
                } else {
                    0.0
                },
                matched_keywords,
            })
            .collect::<Vec<_>>();

        let top_percent = categories
            .iter()
            .find(|category| category.id == config.escalation_category_id)
            .map(|category| category.percent)
            .unwrap_or(0.0);
        let needs_bigger_model = top_percent >= config.escalation_percent_threshold
            || token_estimate >= config.escalation_token_threshold;
        let predicted_subagents = if needs_bigger_model {
            (token_estimate / config.scoring.agents_per_words).clamp(
                config.scoring.min_escalation_agents,
                config.scoring.max_escalation_agents,
            )
        } else {
            config.scoring.normal_agents
        };
        let escalation_catalog = config
            .escalation_catalog
            .iter()
            .map(|target| EscalationCandidate {
                active: request
                    .active_subscriptions
                    .iter()
                    .any(|id| id == &target.subscription_id),
                target: target.clone(),
            })
            .collect();
        let recommended_escalation = if needs_bigger_model {
            select_escalation(
                &config.escalation_catalog,
                &request.active_subscriptions,
                routing_need,
                &config.scoring,
            )
        } else {
            None
        };

        ClassificationSnapshot {
            schema_version: "lfg.classification.v1".into(),
            config_revision: config.revision.clone(),
            config_digest: digest_value(
                &serde_json::to_value(config).expect("classifier configuration serializes"),
            ),
            session_id: request.session_id.clone(),
            revision: request.revision,
            duration_micros: started.elapsed().as_micros() as u64,
            token_estimate,
            categories,
            needs_bigger_model,
            predicted_subagents,
            escalation_catalog,
            recommended_escalation,
            routing_need: routing_need.clone(),
            project: config.project.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphTotals {
    pub predicted: usize,
    pub committed: usize,
    pub queued: usize,
    pub active: usize,
    pub blocked: usize,
    pub completed: usize,
    pub failed: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub plain_label: String,
    pub state: String,
    pub tone: String,
    pub provenance: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relation: String,
    pub state: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphSnapshot {
    pub workflow_id: String,
    pub through_sequence: u64,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub totals: GraphTotals,
}

impl GraphSnapshot {
    pub fn fold(workflow_id: &str, signals: &[SignalEnvelope]) -> Result<Self, ProtocolError> {
        validate_signal_stream(signals)?;
        let mut nodes = BTreeMap::<String, GraphNode>::new();
        let mut edges = BTreeMap::<String, GraphEdge>::new();
        let mut through_sequence = 0;
        for signal in signals
            .iter()
            .filter(|value| value.data.workflow_id == workflow_id)
        {
            through_sequence = signal.data.sequence;
            apply_signal(&mut nodes, &mut edges, signal);
        }
        let mut snapshot = Self {
            workflow_id: workflow_id.to_owned(),
            through_sequence,
            nodes: nodes.into_values().collect(),
            edges: edges.into_values().collect(),
            totals: GraphTotals::default(),
        };
        snapshot.recount();
        Ok(snapshot)
    }

    fn recount(&mut self) {
        for node in &self.nodes {
            if node.kind != "agent" {
                continue;
            }
            match node.state.as_str() {
                "predicted" => self.totals.predicted += 1,
                "committed" => self.totals.committed += 1,
                "queued" => {
                    self.totals.committed += 1;
                    self.totals.queued += 1;
                }
                "active" => {
                    self.totals.committed += 1;
                    self.totals.active += 1;
                }
                "blocked" => {
                    self.totals.committed += 1;
                    self.totals.blocked += 1;
                }
                "completed" => {
                    self.totals.committed += 1;
                    self.totals.completed += 1;
                }
                "failed" => {
                    self.totals.committed += 1;
                    self.totals.failed += 1;
                }
                _ => {}
            }
        }
    }
}

fn apply_signal(
    nodes: &mut BTreeMap<String, GraphNode>,
    edges: &mut BTreeMap<String, GraphEdge>,
    signal: &SignalEnvelope,
) {
    if signal.event_type.ends_with("workflow.started.v1") {
        nodes.insert(
            "prompt".into(),
            GraphNode {
                id: "prompt".into(),
                kind: "prompt".into(),
                label: "PROMPT".into(),
                plain_label: "Your request".into(),
                state: "committed".into(),
                tone: "lime".into(),
                provenance: "live".into(),
                metadata: BTreeMap::new(),
            },
        );
    }
    if signal.event_type.ends_with("plan.committed.v1")
        && let Ok(plan) = serde_json::from_value::<TeamPlan>(signal.data.payload.clone())
    {
        for rro in plan.rros {
            nodes.insert(
                rro.id.clone(),
                GraphNode {
                    id: rro.id.clone(),
                    kind: "rro".into(),
                    label: rro.category.label,
                    plain_label: rro.category.plain_label,
                    state: "committed".into(),
                    tone: rro.category.tone,
                    provenance: "live".into(),
                    metadata: BTreeMap::from([("category".into(), rro.category.id)]),
                },
            );
            let edge_id = format!("prompt->{}", rro.id);
            edges.insert(
                edge_id.clone(),
                GraphEdge {
                    id: edge_id,
                    source: "prompt".into(),
                    target: rro.id,
                    relation: "compiled".into(),
                    state: "committed".into(),
                },
            );
        }
        for agent in plan.agents {
            nodes.insert(
                agent.id.clone(),
                GraphNode {
                    id: agent.id.clone(),
                    kind: "agent".into(),
                    label: agent.role.id.clone(),
                    plain_label: "Specialist agent".into(),
                    state: "committed".into(),
                    tone: "lime".into(),
                    provenance: "live".into(),
                    metadata: BTreeMap::new(),
                },
            );
        }
        for task in plan.tasks {
            nodes.insert(
                task.id.clone(),
                GraphNode {
                    id: task.id.clone(),
                    kind: "task".into(),
                    label: task.kind.to_uppercase(),
                    plain_label: format!("{} task", task.kind),
                    state: "committed".into(),
                    tone: "cyan".into(),
                    provenance: "live".into(),
                    metadata: BTreeMap::from([("sideEffect".into(), task.side_effect_class)]),
                },
            );
            let edge_id = format!("{}->{}", task.agent_id, task.id);
            edges.insert(
                edge_id.clone(),
                GraphEdge {
                    id: edge_id,
                    source: task.agent_id.clone(),
                    target: task.id.clone(),
                    relation: "assigned".into(),
                    state: "committed".into(),
                },
            );
            for rro_id in task.rro_ids {
                let edge_id = format!("{rro_id}->{}", task.id);
                edges.insert(
                    edge_id.clone(),
                    GraphEdge {
                        id: edge_id,
                        source: rro_id,
                        target: task.id.clone(),
                        relation: "requires".into(),
                        state: "committed".into(),
                    },
                );
            }
        }
        for edge in plan.edges {
            let id = format!("{}->{}", edge.from_task_id, edge.to_task_id);
            edges.insert(
                id.clone(),
                GraphEdge {
                    id,
                    source: edge.from_task_id,
                    target: edge.to_task_id,
                    relation: edge.condition,
                    state: "committed".into(),
                },
            );
        }
    }
    if let Some(agent_id) = signal.data.agent_id.as_ref()
        && let Some(node) = nodes.get_mut(agent_id)
    {
        if signal.event_type.ends_with("agent.started.v1") {
            node.state = "active".into();
        }
        if signal.event_type.ends_with("agent.completed.v1") {
            node.state = "completed".into();
        }
        if signal.event_type.ends_with("agent.failed.v1") {
            node.state = "failed".into();
            node.tone = "red".into();
        }
    }
    if signal.event_type.ends_with("route.binding_committed.v1")
        && let Ok(binding) = serde_json::from_value::<ProviderBinding>(signal.data.payload.clone())
    {
        let provider_id = format!("provider:{}", binding.target.target_id);
        nodes.insert(
            provider_id.clone(),
            GraphNode {
                id: provider_id.clone(),
                kind: "provider".into(),
                label: binding.target.display_name,
                plain_label: "Execution provider".into(),
                state: binding.target.availability,
                tone: "cyan".into(),
                provenance: binding.target.provenance,
                metadata: BTreeMap::from([
                    ("provider".into(), binding.target.provider),
                    ("model".into(), binding.target.model),
                ]),
            },
        );
        let id = format!("{}->{}", binding.agent_id, provider_id);
        edges.insert(
            id.clone(),
            GraphEdge {
                id,
                source: binding.agent_id,
                target: provider_id,
                relation: "bound".into(),
                state: "committed".into(),
            },
        );
    }
}

pub fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into())
}

pub fn digest_text(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

pub fn digest_value(value: &Value) -> String {
    let canonical = canonicalize(value);
    let encoded = serde_json::to_vec(&canonical).expect("canonical JSON serializes");
    format!("{:x}", Sha256::digest(encoded))
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted = object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize(value)))
                .collect::<BTreeMap<_, _>>();
            Value::Object(sorted.into_iter().collect::<Map<_, _>>())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonicalize).collect()),
        _ => value.clone(),
    }
}

fn reject_forbidden_content(value: &Value, path: &str) -> Result<(), ProtocolError> {
    const FORBIDDEN: &[&str] = &[
        "prompt",
        "response",
        "content",
        "arguments",
        "result",
        "secret",
        "password",
        "token",
        "api_key",
        "environment",
        "reasoning",
    ];
    match value {
        Value::Object(object) => {
            for (key, nested) in object {
                let normalized = key.to_lowercase();
                if FORBIDDEN.iter().any(|forbidden| {
                    normalized == *forbidden || normalized.ends_with(&format!("_{forbidden}"))
                }) {
                    return Err(ProtocolError::ForbiddenContent(format!("{path}.{key}")));
                }
                reject_forbidden_content(nested, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(values) => {
            for (index, nested) in values.iter().enumerate() {
                reject_forbidden_content(nested, &format!("{path}[{index}]"))?;
            }
        }
        Value::String(text) if looks_like_credential(text) => {
            return Err(ProtocolError::ForbiddenContent(path.to_owned()));
        }
        _ => {}
    }
    Ok(())
}

fn looks_like_credential(text: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "sk-",
        "AKIA",
        "AIza",
        "ya29.",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "-----BEGIN",
    ];
    PREFIXES.iter().any(|prefix| text.starts_with(prefix))
}

pub fn safe_descriptor(label: &str, fingerprint: &str) -> Value {
    json!({"label": label, "fingerprint": fingerprint, "contentPolicy": "metadata-only"})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_hash_chain_and_privacy_are_enforced() {
        let first = SignalEnvelope::seal(
            0,
            None,
            SignalDraft {
                event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
                workflow_id: "workflow-1".into(),
                plan_id: None,
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: None,
                payload_class: "control".into(),
                payload: safe_descriptor("task", "opaque"),
            },
        );
        let second = SignalEnvelope::seal(
            1,
            Some(first.data.event_hash.clone()),
            SignalDraft {
                event_type: "dev.clyffy.lfg.plan.committed.v1".into(),
                workflow_id: "workflow-1".into(),
                plan_id: Some("plan-1".into()),
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: Some(first.id.clone()),
                payload_class: "control".into(),
                payload: json!({"planId": "plan-1"}),
            },
        );
        assert!(validate_signal_stream(&[first.clone(), second]).is_ok());
        let forbidden = SignalEnvelope::seal(
            0,
            None,
            SignalDraft {
                event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
                workflow_id: "workflow-2".into(),
                plan_id: None,
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: None,
                payload_class: "control".into(),
                payload: json!({"prompt": "do not store me"}),
            },
        );
        assert!(matches!(
            forbidden.validate(),
            Err(ProtocolError::ForbiddenContent(_))
        ));
    }

    #[test]
    fn check_continuation_matches_validate_signal_stream() {
        let first = SignalEnvelope::seal(
            0,
            None,
            SignalDraft {
                event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
                workflow_id: "workflow-4".into(),
                plan_id: None,
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: None,
                payload_class: "control".into(),
                payload: safe_descriptor("task", "opaque"),
            },
        );
        assert!(check_continuation(None, &first).is_ok());
        assert!(
            check_continuation(Some((0, "wrong-hash")), &first).is_err(),
            "a signal claiming to be first must not continue an existing tail"
        );

        let second = SignalEnvelope::seal(
            1,
            Some(first.data.event_hash.clone()),
            SignalDraft {
                event_type: "dev.clyffy.lfg.plan.committed.v1".into(),
                workflow_id: "workflow-4".into(),
                plan_id: Some("plan-1".into()),
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: Some(first.id.clone()),
                payload_class: "control".into(),
                payload: json!({"planId": "plan-1"}),
            },
        );
        let tail = (first.data.sequence, first.data.event_hash.as_str());
        assert!(check_continuation(Some(tail), &second).is_ok());
        assert!(
            check_continuation(None, &second).is_err(),
            "a non-zero sequence must not be accepted as a workflow's first signal"
        );
        assert!(validate_signal_stream(&[first, second]).is_ok());
    }

    #[test]
    fn credential_shaped_values_are_rejected_regardless_of_key_name() {
        let signal = SignalEnvelope::seal(
            0,
            None,
            SignalDraft {
                event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
                workflow_id: "workflow-3".into(),
                plan_id: None,
                task_id: None,
                agent_id: None,
                attempt_id: None,
                binding_id: None,
                causation_id: None,
                payload_class: "control".into(),
                payload: json!({"config": "sk-abcdefghijklmnopqrstuvwxyz"}),
            },
        );
        assert!(matches!(
            signal.validate(),
            Err(ProtocolError::ForbiddenContent(_))
        ));
    }

    #[test]
    fn a_grader_cannot_grade_the_outcome_it_produced() {
        let rubric = EvaluationRubric::default_alpha();
        let request = SemanticGradeRequest {
            experiment_id: "exp-1".into(),
            variant_id: "control".into(),
            outcome_id: "run-shared".into(),
            grader_run_id: "run-shared".into(),
            scores: rubric
                .criteria
                .keys()
                .map(|criterion| (criterion.clone(), 0.9))
                .collect(),
        };
        assert!(matches!(
            grade_request(request, &rubric),
            Err(ProtocolError::InvalidPlan(_))
        ));
    }

    #[test]
    fn semantic_statistics_weight_outcomes_equally_and_do_not_infer_independence() {
        let rubric = EvaluationRubric::default_alpha();
        let make_grade = |variant: &str, outcome: &str, grader: String, score: f64| {
            grade_request(
                SemanticGradeRequest {
                    experiment_id: "exp".into(),
                    variant_id: variant.into(),
                    outcome_id: outcome.into(),
                    grader_run_id: grader,
                    scores: rubric
                        .criteria
                        .keys()
                        .map(|key| (key.clone(), score))
                        .collect(),
                },
                &rubric,
            )
            .unwrap()
        };
        let mut grades = (0..9)
            .map(|i| make_grade("candidate", "one", format!("run-{i}"), 1.0))
            .collect::<Vec<_>>();
        grades.push(make_grade("candidate", "two", "run-other".into(), 0.0));
        grades.push(make_grade(
            "baseline",
            "baseline-outcome",
            "baseline-grader".into(),
            0.4,
        ));
        let stats = experiment_statistics("exp", &grades);
        let candidate = stats
            .variants
            .iter()
            .find(|v| v.variant_id == "candidate")
            .unwrap();
        assert_eq!(candidate.samples, 2);
        assert_eq!(candidate.grader_runs, Some(10));
        assert_eq!(candidate.independent_graders, 0);
        assert!((candidate.mean_score - 0.5).abs() < 1e-12);
        assert!((candidate.pass_rate - 0.5).abs() < 1e-12);
        let mut policy = PromotionPolicy::default_alpha();
        policy.minimum_samples_per_arm = 1;
        policy.min_independent_graders = 1;
        assert_eq!(
            promotion_recommendation(&stats, "baseline", "candidate", &policy),
            "insufficient-independent-evidence"
        );
        assert_eq!(
            promotion_recommendation(&stats, "candidate", "candidate", &policy),
            "invalid-comparison"
        );
    }

    #[test]
    fn legacy_statistics_deserialize_without_inventing_grader_run_counts() {
        let historical = json!({
            "variantId":"historical", "samples":1, "independentGraders":2,
            "meanScore":0.5, "passRate":0.5, "failureRate":0.5
        });
        let stats: VariantStatistics = serde_json::from_value(historical.clone()).unwrap();
        assert_eq!(stats.grader_runs, None);
        assert_eq!(serde_json::to_value(stats).unwrap(), historical);
    }

    #[test]
    fn classifier_favors_the_matching_category_and_flags_escalation() {
        let config = ClassifierConfig::default_alpha();
        let snapshot = CpuCategoryClassifier.classify(
            &ClassifyRequest {
                session_id: "s1".into(),
                revision: 1,
                text: "First, refactor the auth module across multiple files, then implement a new session store, and finally migrate the tests step by step.".into(),
                active_subscriptions: vec!["claude".into()],
                routing_need: "balanced".into(),
            },
            &config,
        );
        let top = snapshot
            .categories
            .iter()
            .max_by(|a, b| a.percent.partial_cmp(&b.percent).unwrap())
            .unwrap();
        assert_eq!(top.id, "complex_multi_step_agentic_task");
        assert!(snapshot.needs_bigger_model);
        assert!(snapshot.escalation_catalog.is_empty());
        assert!(snapshot.predicted_subagents >= 2);
        let sum: f64 = snapshot
            .categories
            .iter()
            .map(|category| category.percent)
            .sum();
        assert!((sum - 100.0).abs() < 0.5);

        assert!(snapshot.recommended_escalation.is_none());
    }

    #[test]
    fn classifier_does_not_escalate_a_quick_question() {
        let config = ClassifierConfig::default_alpha();
        let snapshot = CpuCategoryClassifier.classify(
            &ClassifyRequest {
                session_id: "s1".into(),
                revision: 1,
                text: "What is the capital of France?".into(),
                active_subscriptions: vec!["claude".into()],
                routing_need: "balanced".into(),
            },
            &config,
        );
        assert!(!snapshot.needs_bigger_model);
        assert!(snapshot.escalation_catalog.is_empty());
        assert!(snapshot.recommended_escalation.is_none());
        assert_eq!(snapshot.predicted_subagents, 1);
    }

    #[test]
    fn configured_catalog_and_weights_control_selection() {
        let mut config = ClassifierConfig::default_alpha();
        config.escalation_catalog = vec![
            EscalationTarget {
                provider: "example-a".into(),
                model: "a".into(),
                role: "test".into(),
                raw_capability: 0.8,
                tool_loop_efficiency: 0.2,
                cost_tier: "low".into(),
                subscription_id: "a".into(),
                strengths: vec![],
                effort_levels: vec![],
                evidence: ModelProfileEvidence {
                    source: "test fixture".into(),
                    basis: "synthetic".into(),
                    observed_at: "test".into(),
                },
            },
            EscalationTarget {
                provider: "example-b".into(),
                model: "b".into(),
                role: "test".into(),
                raw_capability: 0.2,
                tool_loop_efficiency: 0.8,
                cost_tier: "high".into(),
                subscription_id: "b".into(),
                strengths: vec![],
                effort_levels: vec![],
                evidence: ModelProfileEvidence {
                    source: "test fixture".into(),
                    basis: "synthetic".into(),
                    observed_at: "test".into(),
                },
            },
        ];
        config.validate().unwrap();
        let active = vec!["a".into(), "b".into()];
        config.scoring.capability_weight = 1.0;
        config.scoring.efficiency_weight = 0.0;
        config.scoring.cost_weight = 0.0;
        assert_eq!(
            select_escalation(
                &config.escalation_catalog,
                &active,
                "balanced",
                &config.scoring
            )
            .unwrap()
            .model,
            "a"
        );
        config.scoring.capability_weight = 0.0;
        config.scoring.efficiency_weight = 1.0;
        assert_eq!(
            select_escalation(
                &config.escalation_catalog,
                &active,
                "balanced",
                &config.scoring
            )
            .unwrap()
            .model,
            "b"
        );
        assert!(
            select_escalation(&config.escalation_catalog, &[], "balanced", &config.scoring)
                .is_none()
        );
        assert!(
            select_escalation(
                &config.escalation_catalog,
                &active,
                "unknown",
                &config.scoring
            )
            .is_none()
        );
    }

    #[test]
    fn invalid_classifier_policy_is_rejected() {
        let mut config = ClassifierConfig::default_alpha();
        config.scoring.agents_per_words = 0;
        assert!(config.validate().is_err());
        config.scoring.agents_per_words = 1;
        config.escalation_category_id = "missing".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn classifier_uses_configured_routing_default() {
        let mut config = ClassifierConfig::default_alpha();
        config.default_routing_need = "cost".into();
        let snapshot = CpuCategoryClassifier.classify(
            &ClassifyRequest {
                session_id: "test".into(),
                revision: 1,
                text: "hello".into(),
                active_subscriptions: vec![],
                routing_need: String::new(),
            },
            &config,
        );
        assert_eq!(snapshot.routing_need, "cost");
    }

    #[test]
    fn preflight_reports_measured_dynamic_branches() {
        let snapshot = CpuRuleEncoder::default().encode(&PreflightRequest {
            session_id: "s1".into(),
            revision: 7,
            text: "Research, implement, and test an image workflow".into(),
        });
        assert!(snapshot.predicted_agents >= 3);
        assert!(snapshot.duration_micros < 1_000_000);
        assert!(
            snapshot
                .rros
                .iter()
                .any(|rro| rro.category.capability == "image.generate")
        );
        assert_eq!(snapshot.schema_version, "lfg.preflight.v2");
        assert!(snapshot.confidence.is_none());
        assert!(
            snapshot
                .intents
                .iter()
                .all(|item| item.confidence.is_none())
        );
        assert!(snapshot.rros.iter().all(|item| item.confidence.is_none()));
        assert!(snapshot.abstained);
    }

    #[test]
    fn preflight_configuration_changes_output_and_manifest() {
        let baseline = CpuRuleEncoder::default();
        let mut config = baseline.config.clone();
        config.rules[0].keywords = vec!["custom-visual-signal".into()];
        config.max_branches = 1;
        let customized = CpuRuleEncoder::new(config).unwrap();
        let request = PreflightRequest {
            session_id: "configured".into(),
            revision: 1,
            text: "custom-visual-signal and test".into(),
        };
        let output = customized.encode(&request);
        assert!(
            output
                .intents
                .iter()
                .any(|item| item.label == "create.image")
        );
        assert_eq!(output.predicted_branches, 1);
        assert_ne!(baseline.manifest().sha256, customized.manifest().sha256);

        let mut invalid = customized.config.clone();
        invalid.max_branches = 0;
        assert!(CpuRuleEncoder::new(invalid).is_err());
    }
}
