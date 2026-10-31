use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use lfg_core::{
    AgentSpec, CpuRuleEncoder, EvaluationRubric, ExperimentStatistics, GraphSnapshot, ManifestRef,
    PreflightEncoder, PreflightRequest, PreflightSnapshot, PromotionDecision,
    PromotionDecisionRequest, PromotionPolicy, ProtocolError, ProviderBinding, ProviderTarget,
    SemanticGrade, SemanticGradeRequest, SignalDraft, SignalEnvelope, TaskSpec, TeamPlan,
    check_continuation, digest_text, experiment_statistics, grade_request, now_rfc3339,
    promotion_recommendation, safe_descriptor, validate_signal_stream,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("workflow not found: {0}")]
    WorkflowNotFound(String),
    #[error("signal identity conflicts with a previously committed event: {0}")]
    SignalConflict(String),
    #[error("runtime storage failed: {0}")]
    Storage(String),
    #[error("request is invalid: {0}")]
    InvalidRequest(String),
    #[error("a grade from this grader already exists for the outcome")]
    GradeExists,
    #[error("promotion eligibility gate is not satisfied")]
    PromotionGate,
}

pub trait SignalStore: Send {
    fn append(&mut self, signal: SignalEnvelope) -> Result<(), RuntimeError>;
    fn signals(&self) -> Result<Vec<SignalEnvelope>, RuntimeError>;

    fn workflow(&self, workflow_id: &str) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        let signals = self.signals()?;
        let result = signals
            .into_iter()
            .filter(|signal| signal.data.workflow_id == workflow_id)
            .collect::<Vec<_>>();
        if result.is_empty() {
            return Err(RuntimeError::WorkflowNotFound(workflow_id.to_owned()));
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Default)]
pub struct MemorySignalStore {
    signals: Vec<SignalEnvelope>,
    identities: BTreeMap<(String, String), String>,
    tails: BTreeMap<String, (u64, String)>,
}

impl SignalStore for MemorySignalStore {
    fn append(&mut self, signal: SignalEnvelope) -> Result<(), RuntimeError> {
        signal.validate()?;
        let key = (signal.source.clone(), signal.id.clone());
        if let Some(hash) = self.identities.get(&key) {
            if hash == &signal.data.event_hash {
                return Ok(());
            }
            return Err(RuntimeError::SignalConflict(signal.id));
        }
        let tail = self
            .tails
            .get(&signal.data.workflow_id)
            .map(|(sequence, hash)| (*sequence, hash.as_str()));
        check_continuation(tail, &signal)?;
        self.identities.insert(key, signal.data.event_hash.clone());
        self.tails.insert(
            signal.data.workflow_id.clone(),
            (signal.data.sequence, signal.data.event_hash.clone()),
        );
        self.signals.push(signal);
        Ok(())
    }

    fn signals(&self) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        Ok(self.signals.clone())
    }
}

#[derive(Debug)]
pub struct JsonlSignalStore {
    path: PathBuf,
    signals: Vec<SignalEnvelope>,
    identities: BTreeMap<(String, String), String>,
    tails: BTreeMap<String, (u64, String)>,
}

impl JsonlSignalStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(storage_error)?;
        }
        let mut signals = Vec::new();
        if path.exists() {
            let file = File::open(&path).map_err(storage_error)?;
            for (index, line) in BufReader::new(file).lines().enumerate() {
                let line = line.map_err(storage_error)?;
                if line.trim().is_empty() {
                    continue;
                }
                signals.push(serde_json::from_str(&line).map_err(|error| {
                    RuntimeError::Storage(format!("decode signal line {}: {error}", index + 1))
                })?);
            }
            // One-time O(n) validation of whatever was already on disk. This
            // is not the append-time cost the incremental tail map below
            // exists to avoid — it only runs once, at store-open.
            validate_signal_stream(&signals)?;
        }
        let mut identities = BTreeMap::new();
        let mut tails: BTreeMap<String, (u64, String)> = BTreeMap::new();
        for signal in &signals {
            identities.insert(
                (signal.source.clone(), signal.id.clone()),
                signal.data.event_hash.clone(),
            );
            tails.insert(
                signal.data.workflow_id.clone(),
                (signal.data.sequence, signal.data.event_hash.clone()),
            );
        }
        Ok(Self {
            path,
            signals,
            identities,
            tails,
        })
    }
}

impl SignalStore for JsonlSignalStore {
    fn append(&mut self, signal: SignalEnvelope) -> Result<(), RuntimeError> {
        signal.validate()?;
        let key = (signal.source.clone(), signal.id.clone());
        if let Some(hash) = self.identities.get(&key) {
            if hash == &signal.data.event_hash {
                return Ok(());
            }
            return Err(RuntimeError::SignalConflict(signal.id));
        }
        let tail = self
            .tails
            .get(&signal.data.workflow_id)
            .map(|(sequence, hash)| (*sequence, hash.as_str()));
        check_continuation(tail, &signal)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(storage_error)?;
        serde_json::to_writer(&mut file, &signal).map_err(storage_error)?;
        file.write_all(b"\n").map_err(storage_error)?;
        file.sync_data().map_err(storage_error)?;
        self.identities.insert(key, signal.data.event_hash.clone());
        self.tails.insert(
            signal.data.workflow_id.clone(),
            (signal.data.sequence, signal.data.event_hash.clone()),
        );
        self.signals.push(signal);
        Ok(())
    }

    fn signals(&self) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        Ok(self.signals.clone())
    }
}

#[derive(Debug)]
pub enum RuntimeSignalStore {
    Memory(MemorySignalStore),
    Jsonl(JsonlSignalStore),
}

impl RuntimeSignalStore {
    pub fn memory() -> Self {
        Self::Memory(MemorySignalStore::default())
    }

    pub fn jsonl(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        JsonlSignalStore::open(path).map(Self::Jsonl)
    }
}

impl SignalStore for RuntimeSignalStore {
    fn append(&mut self, signal: SignalEnvelope) -> Result<(), RuntimeError> {
        match self {
            Self::Memory(store) => store.append(signal),
            Self::Jsonl(store) => store.append(signal),
        }
    }

    fn signals(&self) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        match self {
            Self::Memory(store) => store.signals(),
            Self::Jsonl(store) => store.signals(),
        }
    }
}

fn storage_error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Storage(error.to_string())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitPlanRequest {
    pub session_id: String,
    pub revision: u64,
    pub text: String,
    #[serde(default)]
    pub include_fixture_bindings: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanCommit {
    pub workflow_id: String,
    pub plan: TeamPlan,
    pub graph: GraphSnapshot,
    pub committed_signals: usize,
}

#[derive(Clone, Debug)]
pub struct ExecutionMetrics {
    pub input_tokens: i64,
    pub cached_input_tokens: i64,
    pub output_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub response_length: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub name: &'static str,
    pub phase: &'static str,
    pub protocol_version: &'static str,
    pub workspace_id: &'static str,
    pub encoder: ManifestRef,
    pub signal_mode: &'static str,
    pub content_policy: &'static str,
    pub live_providers: usize,
    pub configured_providers: usize,
    pub fixture_providers: usize,
    pub operations: OperationsConfig,
    pub operations_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OperationsConfig {
    schema_version: String,
    revision: String,
    default_listen_address: String,
    benchmark_default_iterations: usize,
    benchmark_max_iterations: usize,
    max_prompt_bytes: usize,
    max_http_body_bytes: usize,
    default_signal_page_size: usize,
    max_signal_page_size: usize,
    event_buffer_capacity: usize,
    stream_keepalive_seconds: u64,
    ui_debounce_ms: u32,
    ui_timing_history_samples: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dev_tools: Option<DevToolsConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DevToolsConfig {
    pub stats_enabled: bool,
    pub trace_capacity: u32,
    pub stats_poll_interval_ms: u32,
    pub debounce_min_ms: u32,
    pub debounce_max_ms: u32,
    pub history_min_samples: u32,
    pub history_max_samples: u32,
}

impl OperationsConfig {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if value.schema_version != "lfg.operations.v1"
            || value.revision.trim().is_empty()
            || value
                .default_listen_address
                .parse::<std::net::SocketAddr>()
                .is_err()
            || value.benchmark_default_iterations == 0
            || value.benchmark_max_iterations < value.benchmark_default_iterations
            || value.max_prompt_bytes == 0
            || value.max_http_body_bytes < value.max_prompt_bytes
            || value.default_signal_page_size == 0
            || value.max_signal_page_size < value.default_signal_page_size
            || value.event_buffer_capacity == 0
            || value
                .event_buffer_capacity
                .checked_next_power_of_two()
                .is_none_or(|n| n > isize::MAX as usize)
            || value.stream_keepalive_seconds == 0
            || value.stream_keepalive_seconds > 9_007_199_254_740_991u64
            || value.ui_debounce_ms == 0
            || value.ui_debounce_ms > i32::MAX as u32
            || value.ui_timing_history_samples == 0
            || std::time::Instant::now()
                .checked_add(std::time::Duration::from_secs(
                    value.stream_keepalive_seconds,
                ))
                .is_none()
            || [
                value.benchmark_default_iterations,
                value.benchmark_max_iterations,
                value.max_prompt_bytes,
                value.max_http_body_bytes,
                value.default_signal_page_size,
                value.max_signal_page_size,
                value.event_buffer_capacity,
            ]
            .iter()
            .any(|n| (*n as u128) > 9_007_199_254_740_991u128)
        {
            return Err("invalid operational configuration".into());
        }
        if let Some(dev) = &value.dev_tools
            && (dev.trace_capacity == 0
                || dev.stats_poll_interval_ms == 0
                || dev.stats_poll_interval_ms > i32::MAX as u32
                || dev.debounce_min_ms == 0
                || dev.debounce_max_ms > i32::MAX as u32
                || !(dev.debounce_min_ms..=dev.debounce_max_ms).contains(&value.ui_debounce_ms)
                || dev.history_min_samples == 0
                || !(dev.history_min_samples..=dev.history_max_samples)
                    .contains(&value.ui_timing_history_samples))
        {
            return Err("invalid development tools configuration".into());
        }
        Ok(value)
    }

    pub fn max_prompt_bytes(&self) -> usize {
        self.max_prompt_bytes
    }
    pub fn default_listen_address(&self) -> std::net::SocketAddr {
        self.default_listen_address
            .parse()
            .expect("validated listen address")
    }
    pub fn benchmark_default_iterations(&self) -> usize {
        self.benchmark_default_iterations
    }
    pub fn benchmark_max_iterations(&self) -> usize {
        self.benchmark_max_iterations
    }
    pub fn max_http_body_bytes(&self) -> usize {
        self.max_http_body_bytes
    }
    pub fn default_signal_page_size(&self) -> usize {
        self.default_signal_page_size
    }
    pub fn max_signal_page_size(&self) -> usize {
        self.max_signal_page_size
    }
    pub fn event_buffer_capacity(&self) -> usize {
        self.event_buffer_capacity
    }
    pub fn stream_keepalive_seconds(&self) -> u64 {
        self.stream_keepalive_seconds
    }

    pub fn dev_tools(&self) -> Option<&DevToolsConfig> {
        self.dev_tools.as_ref()
    }

    pub fn validate_prompt(&self, session_id: &str, text: &str) -> Result<(), RuntimeError> {
        if session_id.trim().is_empty()
            || text.trim().is_empty()
            || text.len() > self.max_prompt_bytes
        {
            return Err(RuntimeError::InvalidRequest(format!(
                "sessionId and nonempty text of at most {} UTF-8 bytes are required",
                self.max_prompt_bytes
            )));
        }
        Ok(())
    }
}

impl Default for OperationsConfig {
    fn default() -> Self {
        Self::from_json_str(include_str!("../config/operations.v1.json"))
            .expect("bundled operational configuration is valid")
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderInventoryConfig {
    schema_version: String,
    revision: String,
    providers: Vec<ProviderTarget>,
}

impl ProviderInventoryConfig {
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if config.schema_version != "lfg.provider-inventory.v1" || config.revision.trim().is_empty()
        {
            return Err("invalid provider inventory schema or revision".into());
        }
        let mut ids = BTreeSet::new();
        for p in &config.providers {
            let fields = [
                &p.target_id,
                &p.display_name,
                &p.provider,
                &p.adapter_id,
                &p.model,
            ];
            if fields.iter().any(|value| value.trim().is_empty())
                || !ids.insert(&p.target_id)
                || p.capabilities.iter().any(|value| value.trim().is_empty())
                || !matches!(
                    (p.provenance.as_str(), p.availability.as_str()),
                    ("configured", "configured") | ("fixture", "replay-only")
                )
            {
                return Err("invalid provider inventory entry".into());
            }
        }
        Ok(config)
    }
}

impl Default for ProviderInventoryConfig {
    fn default() -> Self {
        Self::from_json_str(include_str!("../config/providers.v1.json"))
            .expect("bundled provider inventory is valid")
    }
}

pub struct LfgRuntime<S, E = CpuRuleEncoder> {
    store: S,
    encoder: E,
    providers: Vec<ProviderTarget>,
    rubric: EvaluationRubric,
    promotion_policy: PromotionPolicy,
    operations: OperationsConfig,
    operations_digest: String,
}

impl<S: SignalStore> LfgRuntime<S, CpuRuleEncoder> {
    pub fn new(store: S) -> Self {
        Self::with_encoder(store, CpuRuleEncoder::default())
    }
}

impl<S: SignalStore, E: PreflightEncoder> LfgRuntime<S, E> {
    pub fn with_encoder(store: S, encoder: E) -> Self {
        Self::with_encoder_and_inventory(store, encoder, ProviderInventoryConfig::default())
    }

    pub fn with_encoder_and_inventory(
        store: S,
        encoder: E,
        inventory: ProviderInventoryConfig,
    ) -> Self {
        Self::with_configuration(
            store,
            encoder,
            inventory,
            EvaluationRubric::default_alpha(),
            PromotionPolicy::default_alpha(),
            OperationsConfig::default(),
        )
    }

    pub fn with_configuration(
        store: S,
        encoder: E,
        inventory: ProviderInventoryConfig,
        rubric: EvaluationRubric,
        promotion_policy: PromotionPolicy,
        operations: OperationsConfig,
    ) -> Self {
        let operations_digest = lfg_core::digest_value(
            &serde_json::to_value(&operations).expect("operations serialize"),
        );
        Self {
            store,
            encoder,
            providers: inventory.providers,
            rubric,
            promotion_policy,
            operations,
            operations_digest,
        }
    }

    pub fn status(&self) -> RuntimeStatus {
        RuntimeStatus {
            name: "MasterClyffy LFG",
            phase: "rust-authoritative-alpha",
            protocol_version: lfg_core::PROTOCOL_VERSION,
            workspace_id: lfg_core::MASTER_WORKSPACE_ID,
            encoder: self.encoder.manifest(),
            signal_mode: "validated append-only hash-chain",
            content_policy: "raw prompts are process-memory-only",
            operations: self.operations.clone(),
            operations_digest: self.operations_digest.clone(),
            live_providers: self
                .providers
                .iter()
                .filter(|provider| provider.availability == "active")
                .count(),
            configured_providers: self
                .providers
                .iter()
                .filter(|provider| provider.availability == "configured")
                .count(),
            fixture_providers: self
                .providers
                .iter()
                .filter(|provider| provider.provenance == "fixture")
                .count(),
        }
    }

    pub fn set_codex_availability(&mut self, active: bool) {
        if let Some(provider) = self
            .providers
            .iter_mut()
            .find(|provider| provider.adapter_id == "openai.codex-cli")
        {
            provider.availability = if active { "active" } else { "configured" }.into();
            provider.provenance = if active { "live" } else { "configured" }.into();
        }
    }

    pub fn preflight(&self, request: &PreflightRequest) -> Result<PreflightSnapshot, RuntimeError> {
        self.operations
            .validate_prompt(&request.session_id, &request.text)?;
        Ok(self.encoder.encode(request))
    }

    pub fn commit_plan(&mut self, request: CommitPlanRequest) -> Result<PlanCommit, RuntimeError> {
        let preflight = self.preflight(&PreflightRequest {
            session_id: request.session_id.clone(),
            revision: request.revision,
            text: request.text.clone(),
        })?;
        let workflow_id = format!("workflow-{}", Uuid::new_v4());
        let plan = compile_plan(&workflow_id, &preflight);
        plan.validate()?;

        let started = self.append_draft(SignalDraft {
            event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
            workflow_id: workflow_id.clone(),
            plan_id: None,
            task_id: None,
            agent_id: None,
            attempt_id: None,
            binding_id: None,
            causation_id: None,
            payload_class: "control".into(),
            payload: safe_descriptor("operator-request", &digest_text(&request.text)),
        })?;
        let plan_signal = self.append_draft(SignalDraft {
            event_type: "dev.clyffy.lfg.plan.committed.v1".into(),
            workflow_id: workflow_id.clone(),
            plan_id: Some(plan.id.clone()),
            task_id: None,
            agent_id: None,
            attempt_id: None,
            binding_id: None,
            causation_id: Some(started.id),
            payload_class: "control".into(),
            payload: serde_json::to_value(&plan).expect("plan serializes"),
        })?;

        if request.include_fixture_bindings {
            for (index, agent) in plan.agents.iter().enumerate() {
                let Some(task) = plan.tasks.iter().find(|task| task.agent_id == agent.id) else {
                    continue;
                };
                let Some(target) =
                    select_target(&self.providers, &task.required_capabilities, index)
                else {
                    continue;
                };
                let attempt_id = format!("attempt-{}", Uuid::new_v4());
                let mut binding = ProviderBinding {
                    id: format!("binding-{}", Uuid::new_v4()),
                    plan_id: plan.id.clone(),
                    task_id: task.id.clone(),
                    agent_id: agent.id.clone(),
                    attempt_id: attempt_id.clone(),
                    target,
                    committed_at: now_rfc3339(),
                    hash: String::new(),
                };
                binding.seal();
                self.append_draft(SignalDraft {
                    event_type: "dev.clyffy.lfg.route.binding_committed.v1".into(),
                    workflow_id: workflow_id.clone(),
                    plan_id: Some(plan.id.clone()),
                    task_id: Some(task.id.clone()),
                    agent_id: Some(agent.id.clone()),
                    attempt_id: Some(attempt_id),
                    binding_id: Some(binding.id.clone()),
                    causation_id: Some(plan_signal.id.clone()),
                    payload_class: "control".into(),
                    payload: serde_json::to_value(binding).expect("binding serializes"),
                })?;
            }
        }
        let signals = self.store.workflow(&workflow_id)?;
        let graph = GraphSnapshot::fold(&workflow_id, &signals)?;
        Ok(PlanCommit {
            workflow_id,
            plan,
            graph,
            committed_signals: signals.len(),
        })
    }

    pub fn commit_codex_binding(
        &mut self,
        workflow_id: &str,
        plan: &TeamPlan,
    ) -> Result<ProviderBinding, RuntimeError> {
        let task = plan.tasks.first().ok_or_else(|| {
            RuntimeError::InvalidRequest("the plan has no executable task".into())
        })?;
        let agent = plan
            .agents
            .iter()
            .find(|agent| agent.id == task.agent_id)
            .ok_or_else(|| RuntimeError::InvalidRequest("the task has no matching agent".into()))?;
        let target = self
            .providers
            .iter()
            .find(|provider| {
                provider.adapter_id == "openai.codex-cli" && provider.availability == "active"
            })
            .cloned()
            .ok_or_else(|| {
                RuntimeError::InvalidRequest("the Codex subscription provider is not active".into())
            })?;
        let attempt_id = format!("attempt-{}", Uuid::new_v4());
        let mut binding = ProviderBinding {
            id: format!("binding-{}", Uuid::new_v4()),
            plan_id: plan.id.clone(),
            task_id: task.id.clone(),
            agent_id: agent.id.clone(),
            attempt_id: attempt_id.clone(),
            target,
            committed_at: now_rfc3339(),
            hash: String::new(),
        };
        binding.seal();
        self.append_draft(SignalDraft {
            event_type: "dev.clyffy.lfg.route.binding_committed.v1".into(),
            workflow_id: workflow_id.into(),
            plan_id: Some(plan.id.clone()),
            task_id: Some(task.id.clone()),
            agent_id: Some(agent.id.clone()),
            attempt_id: Some(attempt_id),
            binding_id: Some(binding.id.clone()),
            causation_id: None,
            payload_class: "control".into(),
            payload: serde_json::to_value(&binding).expect("binding serializes"),
        })?;
        Ok(binding)
    }

    pub fn record_execution_started(
        &mut self,
        workflow_id: &str,
        binding: &ProviderBinding,
    ) -> Result<(), RuntimeError> {
        self.append_draft(execution_draft(
            "agent.started",
            workflow_id,
            binding,
            serde_json::json!({"operation": "model.turn", "status": "started"}),
        ))?;
        Ok(())
    }

    pub fn record_execution_event(
        &mut self,
        workflow_id: &str,
        binding: &ProviderBinding,
        event_type: &str,
        operation: &str,
        status: &str,
        native_payload_length: usize,
    ) -> Result<(), RuntimeError> {
        self.append_draft(execution_draft("provider.event_observed", workflow_id, binding, serde_json::json!({"nativeEventType": event_type, "operation": operation, "status": status, "nativePayloadLength": native_payload_length})))?;
        Ok(())
    }

    pub fn record_execution_completed(
        &mut self,
        workflow_id: &str,
        binding: &ProviderBinding,
        metrics: &ExecutionMetrics,
    ) -> Result<(), RuntimeError> {
        self.append_draft(execution_draft("agent.completed", workflow_id, binding, serde_json::json!({"status": "completed", "usage": {"inputTokens": metrics.input_tokens, "cachedInputTokens": metrics.cached_input_tokens, "outputTokens": metrics.output_tokens, "reasoningOutputTokens": metrics.reasoning_output_tokens}, "responseLength": metrics.response_length})))?;
        Ok(())
    }

    pub fn record_execution_failed(
        &mut self,
        workflow_id: &str,
        binding: &ProviderBinding,
        failure_class: &str,
    ) -> Result<(), RuntimeError> {
        self.append_draft(execution_draft(
            "agent.failed",
            workflow_id,
            binding,
            serde_json::json!({"status": "failed", "failureClass": failure_class}),
        ))?;
        Ok(())
    }

    pub fn signals(&self, workflow_id: Option<&str>) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        match workflow_id {
            Some(id) => self.store.workflow(id),
            None => self.store.signals(),
        }
    }

    pub fn replay(
        &self,
        workflow_id: &str,
        after_sequence: u64,
        limit: usize,
    ) -> Result<Vec<SignalEnvelope>, RuntimeError> {
        Ok(self
            .store
            .workflow(workflow_id)?
            .into_iter()
            .filter(|signal| signal.data.sequence > after_sequence)
            .take(limit.clamp(1, self.operations.max_signal_page_size))
            .collect())
    }

    pub fn graph(
        &self,
        workflow_id: &str,
        through_sequence: Option<u64>,
    ) -> Result<GraphSnapshot, RuntimeError> {
        let mut signals = self.store.workflow(workflow_id)?;
        if let Some(sequence) = through_sequence {
            signals.retain(|signal| signal.data.sequence <= sequence);
        }
        GraphSnapshot::fold(workflow_id, &signals).map_err(Into::into)
    }

    pub fn rubric(&self) -> EvaluationRubric {
        self.rubric.clone()
    }

    pub fn promotion_policy(&self) -> PromotionPolicy {
        self.promotion_policy.clone()
    }

    pub fn record_grade(
        &mut self,
        request: SemanticGradeRequest,
    ) -> Result<SemanticGrade, RuntimeError> {
        let existing = self.grades(Some(&request.experiment_id))?;
        if existing
            .iter()
            .any(|grade| grade.rubric.sha256 != self.rubric.hash)
        {
            return Err(RuntimeError::InvalidRequest(
                "experiment contains grades from another rubric revision".into(),
            ));
        }
        if existing.iter().any(|grade| {
            grade.outcome_id == request.outcome_id && grade.variant_id != request.variant_id
        }) {
            return Err(RuntimeError::InvalidRequest(
                "an outcome cannot belong to multiple variants in one experiment".into(),
            ));
        }
        if existing.iter().any(|grade| {
            grade.outcome_id == request.outcome_id && grade.grader_run_id == request.grader_run_id
        }) {
            return Err(RuntimeError::GradeExists);
        }
        let grade = grade_request(request, &self.rubric())?;
        self.append_draft(SignalDraft {
            event_type: "dev.clyffy.lfg.evaluation.grade_recorded.v1".into(),
            workflow_id: format!("experiment-{}", grade.experiment_id),
            plan_id: None,
            task_id: None,
            agent_id: None,
            attempt_id: None,
            binding_id: None,
            causation_id: None,
            payload_class: "evaluation".into(),
            payload: serde_json::to_value(&grade).expect("grade serializes"),
        })?;
        Ok(grade)
    }

    pub fn grades(&self, experiment_id: Option<&str>) -> Result<Vec<SemanticGrade>, RuntimeError> {
        let mut result = Vec::new();
        for signal in self
            .store
            .signals()?
            .into_iter()
            .filter(|signal| signal.event_type.ends_with("evaluation.grade_recorded.v1"))
        {
            let grade: SemanticGrade =
                serde_json::from_value(signal.data.payload).map_err(storage_error)?;
            if experiment_id.is_none_or(|id| grade.experiment_id == id) {
                result.push(grade);
            }
        }
        Ok(result)
    }

    pub fn statistics(&self, experiment_id: &str) -> Result<ExperimentStatistics, RuntimeError> {
        let grades = self.grades(Some(experiment_id))?;
        if grades
            .iter()
            .any(|grade| grade.rubric.sha256 != self.rubric.hash)
        {
            return Err(RuntimeError::InvalidRequest(
                "experiment contains grades from another rubric revision".into(),
            ));
        }
        let mut outcomes = BTreeMap::new();
        for grade in &grades {
            if outcomes
                .insert(&grade.outcome_id, &grade.variant_id)
                .is_some_and(|previous| previous != &grade.variant_id)
            {
                return Err(RuntimeError::InvalidRequest(
                    "an outcome belongs to multiple variants in the experiment".into(),
                ));
            }
        }
        Ok(experiment_statistics(experiment_id, &grades))
    }

    pub fn decide_promotion(
        &mut self,
        request: PromotionDecisionRequest,
    ) -> Result<PromotionDecision, RuntimeError> {
        if request.baseline_variant_id.trim().is_empty()
            || request.candidate_variant_id.trim().is_empty()
            || request.baseline_variant_id == request.candidate_variant_id
        {
            return Err(RuntimeError::InvalidRequest(
                "baseline and candidate must be distinct nonempty variants".into(),
            ));
        }
        let statistics = self.statistics(&request.experiment_id)?;
        let policy = self.promotion_policy();
        let recommendation = promotion_recommendation(
            &statistics,
            &request.baseline_variant_id,
            &request.candidate_variant_id,
            &policy,
        );
        if request.action == "promote" && recommendation != "promote" {
            return Err(RuntimeError::PromotionGate);
        }
        if !["promote", "hold", "reject"].contains(&request.action.as_str()) {
            return Err(RuntimeError::InvalidRequest(
                "action must be promote, hold, or reject".into(),
            ));
        }
        let mut decision = PromotionDecision {
            schema_version: "lfg.promotion-decision.v1".into(),
            id: format!("promotion-{}", Uuid::new_v4()),
            experiment_id: request.experiment_id,
            baseline_variant_id: request.baseline_variant_id,
            candidate_variant_id: request.candidate_variant_id,
            policy,
            statistics,
            recommendation,
            action: request.action,
            decided_by: request.decided_by,
            reason_codes: request.reason_codes,
            decided_at: now_rfc3339(),
            hash: String::new(),
        };
        decision.hash =
            lfg_core::digest_value(&serde_json::to_value(&decision).expect("decision serializes"));
        self.append_draft(SignalDraft {
            event_type: "dev.clyffy.lfg.evaluation.promotion_decided.v1".into(),
            workflow_id: format!("experiment-{}", decision.experiment_id),
            plan_id: None,
            task_id: None,
            agent_id: None,
            attempt_id: None,
            binding_id: None,
            causation_id: None,
            payload_class: "evaluation".into(),
            payload: serde_json::to_value(&decision).expect("decision serializes"),
        })?;
        Ok(decision)
    }

    pub fn promotion_decisions(
        &self,
        experiment_id: Option<&str>,
    ) -> Result<Vec<PromotionDecision>, RuntimeError> {
        let mut result = Vec::new();
        for signal in self.store.signals()?.into_iter().filter(|signal| {
            signal
                .event_type
                .ends_with("evaluation.promotion_decided.v1")
        }) {
            let decision: PromotionDecision =
                serde_json::from_value(signal.data.payload).map_err(storage_error)?;
            if experiment_id.is_none_or(|id| decision.experiment_id == id) {
                result.push(decision);
            }
        }
        Ok(result)
    }

    fn append_draft(&mut self, draft: SignalDraft) -> Result<SignalEnvelope, RuntimeError> {
        let existing = self.store.signals()?;
        let tail = existing
            .iter()
            .filter(|signal| signal.data.workflow_id == draft.workflow_id)
            .max_by_key(|signal| signal.data.sequence);
        let (sequence, previous_hash) = tail
            .map(|signal| {
                (
                    signal.data.sequence + 1,
                    Some(signal.data.event_hash.clone()),
                )
            })
            .unwrap_or((0, None));
        let signal = SignalEnvelope::seal(sequence, previous_hash, draft);
        self.store.append(signal.clone())?;
        Ok(signal)
    }
}

fn compile_plan(workflow_id: &str, preflight: &PreflightSnapshot) -> TeamPlan {
    let rros = preflight
        .rros
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, mut rro)| {
            rro.id = format!(
                "rro-{}-{index}",
                &workflow_id[workflow_id.len().saturating_sub(12)..]
            );
            rro.advisory = false;
            rro.lineage = vec![format!("workflow/{workflow_id}")];
            rro
        })
        .collect::<Vec<_>>();
    let actionable = rros
        .iter()
        .filter(|rro| !rro.category.id.starts_with("policy."))
        .collect::<Vec<_>>();
    let mut agents = Vec::new();
    let mut tasks = Vec::new();
    for (index, rro) in actionable.iter().enumerate() {
        let agent_id = format!(
            "agent-{}-{index}",
            &workflow_id[workflow_id.len().saturating_sub(8)..]
        );
        agents.push(AgentSpec {
            id: agent_id.clone(),
            role: ManifestRef::new(
                format!("role-{}", rro.category.id),
                "1.0.0",
                &rro.category.plain_label,
            ),
            required_capabilities: vec![rro.category.capability.clone()],
            max_parallel_tasks: 1,
        });
        tasks.push(TaskSpec {
            id: format!(
                "task-{}-{index}",
                &workflow_id[workflow_id.len().saturating_sub(8)..]
            ),
            kind: task_kind(&rro.category.id),
            agent_id,
            rro_ids: vec![rro.id.clone()],
            required_capabilities: vec![rro.category.capability.clone()],
            side_effect_class: if rro.category.capability == "filesystem.write" {
                "write".into()
            } else {
                "read".into()
            },
        });
    }
    if agents.is_empty() {
        agents.push(AgentSpec {
            id: "agent-general".into(),
            role: ManifestRef::new("role-general", "1.0.0", "general response"),
            required_capabilities: vec!["text.generate".into()],
            max_parallel_tasks: 1,
        });
        tasks.push(TaskSpec {
            id: "task-general".into(),
            kind: "respond".into(),
            agent_id: "agent-general".into(),
            rro_ids: rros
                .first()
                .map(|rro| vec![rro.id.clone()])
                .unwrap_or_default(),
            required_capabilities: vec!["text.generate".into()],
            side_effect_class: "none".into(),
        });
    }
    let mut plan = TeamPlan {
        schema_version: "lfg.team-plan.v1".into(),
        id: format!("plan-{}", Uuid::new_v4()),
        revision: 1,
        workspace_id: lfg_core::MASTER_WORKSPACE_ID.into(),
        project_key: lfg_core::MASTER_PROJECT_KEY.into(),
        compiler: ManifestRef::new(
            "lfg-rro-compiler",
            "1.0.0",
            "CPU preflight to provider-neutral RRO DAG",
        ),
        policy: ManifestRef::new(
            "lfg-metadata-only-policy",
            "1.0.0",
            "no raw content in control ledger",
        ),
        rros,
        agents,
        tasks,
        edges: Vec::new(),
        max_parallel_tasks: preflight.predicted_branches.max(1) as u32,
        created_at: now_rfc3339(),
        hash: String::new(),
    };
    plan.seal();
    plan
}

fn task_kind(category: &str) -> String {
    category.split('.').next().unwrap_or("respond").to_owned()
}

fn select_target(
    providers: &[ProviderTarget],
    required: &[String],
    offset: usize,
) -> Option<ProviderTarget> {
    let eligible = providers
        .iter()
        .filter(|provider| {
            provider.provenance == "fixture"
                && required
                    .iter()
                    .all(|required| provider.capabilities.contains(required))
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        None
    } else {
        Some((*eligible[offset % eligible.len()]).clone())
    }
}

fn execution_draft(
    suffix: &str,
    workflow_id: &str,
    binding: &ProviderBinding,
    payload: serde_json::Value,
) -> SignalDraft {
    SignalDraft {
        event_type: format!("dev.clyffy.lfg.{suffix}.v1"),
        workflow_id: workflow_id.into(),
        plan_id: Some(binding.plan_id.clone()),
        task_id: Some(binding.task_id.clone()),
        agent_id: Some(binding.agent_id.clone()),
        attempt_id: Some(binding.attempt_id.clone()),
        binding_id: Some(binding.id.clone()),
        causation_id: None,
        payload_class: "telemetry".into(),
        payload,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn operations_config_validates_limits_and_enforces_utf8_bytes_and_page_size() {
        let mut settings = serde_json::to_value(OperationsConfig::default()).unwrap();
        let mut legacy = settings.clone();
        legacy.as_object_mut().unwrap().remove("devTools");
        assert!(
            OperationsConfig::from_json_str(&legacy.to_string())
                .unwrap()
                .dev_tools()
                .is_none()
        );
        let mut invalid_dev = settings.clone();
        invalid_dev["devTools"]["debounceMinMs"] = serde_json::json!(0);
        assert!(OperationsConfig::from_json_str(&invalid_dev.to_string()).is_err());
        settings["maxPromptBytes"] = serde_json::json!(3);
        settings["defaultSignalPageSize"] = serde_json::json!(1);
        settings["maxSignalPageSize"] = serde_json::json!(1);
        let operations = OperationsConfig::from_json_str(&settings.to_string()).unwrap();
        let mut runtime = LfgRuntime::with_configuration(
            MemorySignalStore::default(),
            CpuRuleEncoder::default(),
            ProviderInventoryConfig::default(),
            EvaluationRubric::default_alpha(),
            PromotionPolicy::default_alpha(),
            operations,
        );
        let request = |text: &str| PreflightRequest {
            session_id: "test".into(),
            revision: 1,
            text: text.into(),
        };
        assert!(runtime.preflight(&request("é")).is_ok());
        assert!(runtime.preflight(&request("éé")).is_err());
        let plan = runtime
            .commit_plan(CommitPlanRequest {
                session_id: "test".into(),
                revision: 1,
                text: "é".into(),
                include_fixture_bindings: false,
            })
            .unwrap();
        assert!(plan.committed_signals >= 2);
        assert_eq!(runtime.replay(&plan.workflow_id, 0, 100).unwrap().len(), 1);
        settings["maxSignalPageSize"] = serde_json::json!(0);
        assert!(OperationsConfig::from_json_str(&settings.to_string()).is_err());
        settings["maxSignalPageSize"] = serde_json::json!(1);
        settings["eventBufferCapacity"] = serde_json::json!(0);
        assert!(OperationsConfig::from_json_str(&settings.to_string()).is_err());
    }

    #[test]
    fn provider_inventory_is_configurable_and_does_not_forge_bindings() {
        let empty = ProviderInventoryConfig::from_json_str(
            r#"{"schemaVersion":"lfg.provider-inventory.v1","revision":"project-a","providers":[]}"#,
        ).unwrap();
        let runtime = LfgRuntime::with_encoder_and_inventory(
            MemorySignalStore::default(),
            CpuRuleEncoder::default(),
            empty,
        );
        assert_eq!(runtime.status().fixture_providers, 0);
        assert!(select_target(&runtime.providers, &["image.generate".into()], 0).is_none());
        let fixture = ProviderInventoryConfig::default();
        assert!(select_target(&fixture.providers, &["unknown.capability".into()], 0).is_none());
        let mut spoofed = fixture;
        spoofed.providers[0].availability = "active".into();
        assert!(
            ProviderInventoryConfig::from_json_str(&serde_json::to_string(&spoofed).unwrap())
                .is_err()
        );
    }

    #[test]
    fn changed_rubric_cannot_mix_with_existing_experiment_grades() {
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        let scores = runtime
            .rubric()
            .criteria
            .keys()
            .map(|key| (key.clone(), 0.9))
            .collect();
        runtime
            .record_grade(SemanticGradeRequest {
                experiment_id: "experiment".into(),
                variant_id: "a".into(),
                outcome_id: "outcome".into(),
                grader_run_id: "grader".into(),
                scores,
            })
            .unwrap();
        let mut changed = runtime.rubric();
        changed.pass_threshold = 0.95;
        runtime.rubric =
            EvaluationRubric::from_json_str(&serde_json::to_string(&changed).unwrap()).unwrap();
        assert!(runtime.statistics("experiment").is_err());
        let scores = runtime
            .rubric()
            .criteria
            .keys()
            .map(|key| (key.clone(), 0.9))
            .collect();
        assert!(
            runtime
                .record_grade(SemanticGradeRequest {
                    experiment_id: "experiment".into(),
                    variant_id: "b".into(),
                    outcome_id: "other".into(),
                    grader_run_id: "grader".into(),
                    scores
                })
                .is_err()
        );
    }

    #[test]
    fn incremental_append_rejects_bad_continuation_and_tracks_tails_per_workflow() {
        let mut store = MemorySignalStore::default();

        let draft = |workflow_id: &str| SignalDraft {
            event_type: "dev.clyffy.lfg.workflow.started.v1".into(),
            workflow_id: workflow_id.into(),
            plan_id: None,
            task_id: None,
            agent_id: None,
            attempt_id: None,
            binding_id: None,
            causation_id: None,
            payload_class: "control".into(),
            payload: lfg_core::safe_descriptor("task", "opaque"),
        };

        // Two workflows' signals interleaved must not cross-contaminate
        // each other's tracked tail.
        let a0 = SignalEnvelope::seal(0, None, draft("workflow-a"));
        store.append(a0.clone()).unwrap();
        let b0 = SignalEnvelope::seal(0, None, draft("workflow-b"));
        store.append(b0.clone()).unwrap();
        let a1 = SignalEnvelope::seal(1, Some(a0.data.event_hash.clone()), draft("workflow-a"));
        store.append(a1.clone()).unwrap();
        let b1 = SignalEnvelope::seal(1, Some(b0.data.event_hash.clone()), draft("workflow-b"));
        store.append(b1).unwrap();

        // A signal that skips a sequence number must be rejected, not
        // silently accepted because the incremental check only compares
        // against the tracked tail rather than the full history.
        let bad = SignalEnvelope::seal(5, Some(a1.data.event_hash.clone()), draft("workflow-a"));
        assert!(matches!(
            store.append(bad),
            Err(RuntimeError::Protocol(ProtocolError::InvalidContinuation))
        ));

        assert_eq!(store.workflow("workflow-a").unwrap().len(), 2);
        assert_eq!(store.workflow("workflow-b").unwrap().len(), 2);
    }

    #[test]
    fn committed_tallies_are_derived_from_replay() {
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        let result = runtime
            .commit_plan(CommitPlanRequest {
                session_id: "test".into(),
                revision: 1,
                text: "Research, implement, test, and grade an image workflow".into(),
                include_fixture_bindings: true,
            })
            .unwrap();
        assert!(result.graph.totals.committed >= 4);
        assert_eq!(result.graph.totals.predicted, 0);
        assert!(
            result
                .graph
                .nodes
                .iter()
                .any(|node| node.kind == "provider" && node.provenance == "fixture")
        );
        let replayed = runtime.graph(&result.workflow_id, None).unwrap();
        assert_eq!(result.graph, replayed);
    }

    #[test]
    fn jsonl_store_reopens_and_validates_hash_chain() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("signals.jsonl");
        let mut runtime = LfgRuntime::new(JsonlSignalStore::open(&path).unwrap());
        let result = runtime
            .commit_plan(CommitPlanRequest {
                session_id: "test".into(),
                revision: 1,
                text: "implement and test".into(),
                include_fixture_bindings: false,
            })
            .unwrap();
        drop(runtime);
        let reopened = LfgRuntime::new(JsonlSignalStore::open(&path).unwrap());
        assert_eq!(
            reopened.graph(&result.workflow_id, None).unwrap(),
            result.graph
        );
        let contents = fs::read_to_string(path).unwrap();
        assert!(!contents.contains("implement and test"));
    }

    #[test]
    fn explicit_promotion_cannot_bypass_sample_gate() {
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        for (variant, score) in [("control", 0.70), ("candidate", 0.95)] {
            for grader in ["grader-a", "grader-b"] {
                runtime
                    .record_grade(SemanticGradeRequest {
                        experiment_id: "exp-gate".into(),
                        variant_id: variant.into(),
                        outcome_id: format!("{variant}-outcome"),
                        grader_run_id: grader.into(),
                        scores: BTreeMap::from([
                            ("correctness".into(), score),
                            ("evidence".into(), score),
                            ("completeness".into(), score),
                            ("safety".into(), score),
                        ]),
                    })
                    .unwrap();
            }
        }
        let request = PromotionDecisionRequest {
            experiment_id: "exp-gate".into(),
            baseline_variant_id: "control".into(),
            candidate_variant_id: "candidate".into(),
            action: "promote".into(),
            decided_by: ManifestRef::new("operator", "1", "test operator"),
            reason_codes: vec!["test".into()],
        };
        assert!(matches!(
            runtime.decide_promotion(request),
            Err(RuntimeError::PromotionGate)
        ));
    }

    #[test]
    fn grader_run_ids_do_not_satisfy_independence_and_outcomes_cannot_cross_variants() {
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        runtime.promotion_policy.minimum_samples_per_arm = 1;
        runtime.promotion_policy.min_independent_graders = 2;
        for (variant, score) in [("baseline", 0.70), ("candidate", 0.95)] {
            for grader in ["run-a", "run-b"] {
                runtime
                    .record_grade(SemanticGradeRequest {
                        experiment_id: "exp-independence".into(),
                        variant_id: variant.into(),
                        outcome_id: format!("{variant}-outcome"),
                        grader_run_id: grader.into(),
                        scores: runtime
                            .rubric()
                            .criteria
                            .keys()
                            .map(|key| (key.clone(), score))
                            .collect(),
                    })
                    .unwrap();
            }
        }
        let stats = runtime.statistics("exp-independence").unwrap();
        assert!(
            stats
                .variants
                .iter()
                .all(|v| v.grader_runs == Some(2) && v.independent_graders == 0)
        );
        let conflict = SemanticGradeRequest {
            experiment_id: "exp-independence".into(),
            variant_id: "candidate".into(),
            outcome_id: "baseline-outcome".into(),
            grader_run_id: "different-run".into(),
            scores: runtime
                .rubric()
                .criteria
                .keys()
                .map(|key| (key.clone(), 0.9))
                .collect(),
        };
        assert!(matches!(
            runtime.record_grade(conflict),
            Err(RuntimeError::InvalidRequest(_))
        ));
        let request = PromotionDecisionRequest {
            experiment_id: "exp-independence".into(),
            baseline_variant_id: "baseline".into(),
            candidate_variant_id: "candidate".into(),
            action: "promote".into(),
            decided_by: ManifestRef::new("operator", "1", "test operator"),
            reason_codes: vec![],
        };
        assert!(matches!(
            runtime.decide_promotion(request.clone()),
            Err(RuntimeError::PromotionGate)
        ));
        let same = PromotionDecisionRequest {
            candidate_variant_id: "baseline".into(),
            action: "hold".into(),
            ..request
        };
        assert!(matches!(
            runtime.decide_promotion(same),
            Err(RuntimeError::InvalidRequest(_))
        ));
    }

    #[test]
    fn statistics_reject_legacy_outcomes_assigned_to_multiple_variants() {
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        let rubric = runtime.rubric();
        for variant in ["baseline", "candidate"] {
            let grade = grade_request(
                SemanticGradeRequest {
                    experiment_id: "legacy-exp".into(),
                    variant_id: variant.into(),
                    outcome_id: "shared-outcome".into(),
                    grader_run_id: format!("{variant}-grader"),
                    scores: rubric
                        .criteria
                        .keys()
                        .map(|key| (key.clone(), 0.9))
                        .collect(),
                },
                &rubric,
            )
            .unwrap();
            runtime
                .append_draft(SignalDraft {
                    event_type: "dev.clyffy.lfg.evaluation.grade_recorded.v1".into(),
                    workflow_id: "experiment-legacy-exp".into(),
                    plan_id: None,
                    task_id: None,
                    agent_id: None,
                    attempt_id: None,
                    binding_id: None,
                    causation_id: None,
                    payload_class: "evaluation".into(),
                    payload: serde_json::to_value(grade).unwrap(),
                })
                .unwrap();
        }
        assert!(matches!(
            runtime.statistics("legacy-exp"),
            Err(RuntimeError::InvalidRequest(_))
        ));
    }

    #[test]
    fn shared_shadow_fixture_matches_authoritative_projection() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/lfg/shadow-parity.v1.json");
        let fixture: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let prompt = fixture["prompt"].as_str().unwrap();
        let mut runtime = LfgRuntime::new(MemorySignalStore::default());
        let commit = runtime
            .commit_plan(CommitPlanRequest {
                session_id: "shadow-parity".into(),
                revision: 1,
                text: prompt.into(),
                include_fixture_bindings: false,
            })
            .unwrap();
        let expected = &fixture["expected"];
        assert_eq!(commit.plan.workspace_id, expected["workspaceId"]);
        assert_eq!(
            commit.plan.agents.len() as u64,
            expected["agentCount"].as_u64().unwrap()
        );
        assert_eq!(
            commit.plan.tasks.len() as u64,
            expected["taskCount"].as_u64().unwrap()
        );
        let mut capabilities = commit
            .plan
            .tasks
            .iter()
            .flat_map(|task| task.required_capabilities.clone())
            .collect::<Vec<_>>();
        capabilities.sort();
        capabilities.dedup();
        assert_eq!(
            serde_json::to_value(capabilities).unwrap(),
            expected["capabilities"]
        );
        let agents = serde_json::to_string(&commit.plan.agents)
            .unwrap()
            .to_lowercase();
        for provider in ["openai", "anthropic", "gemini", "grok", "cursor"] {
            assert!(!agents.contains(provider));
        }
        assert!(
            !serde_json::to_string(&runtime.signals(Some(&commit.workflow_id)).unwrap())
                .unwrap()
                .contains(prompt)
        );
    }
}
