//! Opt-in, cooperating-writer local journal for kernel observations and corrections.
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::Digest;

const SCHEMA: &str = "lfg.kernel-journal.v1";

#[derive(Deserialize)]
struct SourceRow {
    id: String,
    text: String,
}

pub struct SourceIndex {
    path: PathBuf,
    max_bytes: u64,
    sha256: String,
    rows: BTreeMap<String, String>,
}

impl SourceIndex {
    pub fn read(path: &Path, max_bytes: u64) -> Result<Self, String> {
        if max_bytes == 0 {
            return Err("source size limit must be positive".into());
        }
        let mut file = File::open(path).map_err(|error| format!("cannot open source: {error}"))?;
        let before = file.metadata().map_err(|error| error.to_string())?;
        if !before.is_file() || before.len() > max_bytes {
            return Err("source is not a bounded regular file".into());
        }
        let mut raw = Vec::new();
        (&mut file)
            .take(max_bytes.saturating_add(1))
            .read_to_end(&mut raw)
            .map_err(|error| error.to_string())?;
        let after = file.metadata().map_err(|error| error.to_string())?;
        if raw.len() as u64 != before.len()
            || before.len() != after.len()
            || before.modified().ok() != after.modified().ok()
        {
            return Err("source changed while reading".into());
        }
        let mut rows = BTreeMap::new();
        for line in raw
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        {
            let row: SourceRow =
                serde_json::from_slice(line).map_err(|_| "invalid source JSONL record")?;
            if row.id.trim().is_empty()
                || row.text.trim().is_empty()
                || rows.insert(row.id, row.text).is_some()
            {
                return Err("source IDs must be unique and source text nonempty".into());
            }
        }
        if rows.is_empty() {
            return Err("source contains no records".into());
        }
        Ok(Self {
            path: path.to_owned(),
            max_bytes,
            sha256: format!("{:x}", sha2::Sha256::digest(&raw)),
            rows,
        })
    }

    fn verify_unchanged(&self) -> Result<(), String> {
        if Self::read(&self.path, self.max_bytes)?.sha256 != self.sha256 {
            return Err("source changed during operation".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChoiceObservation {
    node: String,
    option: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Observation {
    request_id: String,
    input_sha256: String,
    captured_text: Option<String>,
    snapshot_sha256: String,
    snapshot_revision: String,
    address: String,
    state: String,
    at_node: String,
    choices: Vec<ChoiceObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Event {
    Observation {
        observation: Observation,
    },
    Correction {
        observation_sequence: u64,
        snapshot_sha256: String,
        node: String,
        choice: String,
        reviewer: String,
        attested: bool,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    schema_version: String,
    sequence: u64,
    previous_sha256: Option<String>,
    event: Event,
    sha256: String,
}

pub trait Journal {
    fn observe(
        &mut self,
        id: &str,
        text: &str,
        result: &lfg_core::kernel::ResultRecord,
        capture_text: bool,
    ) -> Result<u64, String>;
    fn correct(
        &mut self,
        engine: &lfg_core::kernel::Engine,
        observation_sequence: u64,
        node: &str,
        choice: &str,
        reviewer: &str,
    ) -> Result<u64, String>;
}

pub struct JsonlJournal {
    file: File,
    max_bytes: u64,
    length: u64,
    sequence: u64,
    tail: Option<String>,
    observations: BTreeMap<u64, Observation>,
    corrections: Vec<(u64, Event)>,
}

fn digest(sequence: u64, previous: &Option<String>, event: &Event) -> String {
    lfg_core::digest_value(&json!({"schemaVersion":SCHEMA,"sequence":sequence,
        "previousSha256":previous,"event":event}))
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_observation(item: &Observation) -> bool {
    !item.request_id.trim().is_empty()
        && is_digest(&item.input_sha256)
        && is_digest(&item.snapshot_sha256)
        && !item.snapshot_revision.trim().is_empty()
        && !item.address.trim().is_empty()
        && matches!(item.state.as_str(), "complete" | "pruned" | "escalate")
        && !item.at_node.trim().is_empty()
        && item
            .captured_text
            .as_ref()
            .is_none_or(|text| lfg_core::digest_text(text) == item.input_sha256)
        && item
            .choices
            .iter()
            .all(|choice| !choice.node.trim().is_empty() && !choice.option.trim().is_empty())
}

impl JsonlJournal {
    pub fn status(&self) -> Value {
        json!({"schemaVersion":SCHEMA,"entries":self.sequence,
            "observations":self.observations.len(),
            "corrections":self.sequence - self.observations.len() as u64,
            "capturedText":self.observations.values().filter(|item| item.captured_text.is_some()).count(),
            "bytes":self.length,"tailSha256":self.tail})
    }

    pub fn reconcile(&self, source: &SourceIndex) -> Result<Value, String> {
        let mut matched = 0u64;
        let mut missing = 0u64;
        for observation in self.observations.values() {
            if let Some(text) = source.rows.get(&observation.request_id) {
                if lfg_core::digest_text(text) != observation.input_sha256
                    || observation
                        .captured_text
                        .as_ref()
                        .is_some_and(|captured| captured != text)
                {
                    return Err("source text differs from its journal observation".into());
                }
                matched += 1;
            } else {
                missing += 1;
            }
        }
        source.verify_unchanged()?;
        Ok(
            json!({"schemaVersion":"lfg.kernel-source-reconciliation.v1",
            "observations":self.observations.len(),"matched":matched,"missing":missing,
            "sourceRecords":source.rows.len(),"sourceSha256":source.sha256,
            "journalTailSha256":self.tail,"reviewed":false}),
        )
    }

    pub fn show(
        &self,
        engine: &lfg_core::kernel::Engine,
        sequence: u64,
        include_text: bool,
        source: Option<&SourceIndex>,
    ) -> Result<Value, String> {
        let mut observation = self
            .observations
            .get(&sequence)
            .ok_or("unknown observation")?
            .clone();
        if observation.snapshot_sha256 != engine.snapshot_digest() {
            return Err("observation does not match the supplied snapshot".into());
        }
        if let Some(source) = source {
            self.reconcile(source)?;
        }
        let source_text = source.and_then(|index| index.rows.get(&observation.request_id));
        let captured_text_available = observation.captured_text.is_some();
        if !include_text {
            observation.captured_text = None;
        }
        let mut choices = BTreeMap::new();
        for node in observation
            .choices
            .iter()
            .map(|item| item.node.as_str())
            .chain(std::iter::once(observation.at_node.as_str()))
        {
            if let Some(options) = engine.choice_options(node) {
                choices.insert(node.to_owned(), options);
            }
        }
        Ok(
            json!({"observationSequence":sequence,"observation":observation,
            "capturedTextAvailable":captured_text_available,"sourceTextAvailable":source_text.is_some(),
            "sourceText":if include_text { source_text } else { None },
            "sourceFileSha256":source.map(|value| &value.sha256),"choiceOptions":choices}),
        )
    }

    pub fn open(path: &Path, max_bytes: u64, create: bool) -> Result<Self, String> {
        if max_bytes == 0 {
            return Err("journal size limit must be positive".into());
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .truncate(false)
            .open(path)
            .map_err(|error| format!("cannot open journal: {error}"))?;
        file.try_lock()
            .map_err(|error| format!("journal is locked or cannot be locked: {error}"))?;
        let before = file.metadata().map_err(|error| error.to_string())?.len();
        if before > max_bytes {
            return Err("journal exceeds configured size limit".into());
        }
        let mut raw = Vec::new();
        (&mut file)
            .take(max_bytes.saturating_add(1))
            .read_to_end(&mut raw)
            .map_err(|error| error.to_string())?;
        if raw.len() as u64 != before
            || file.metadata().map_err(|error| error.to_string())?.len() != before
            || (!raw.is_empty()
                && (!raw.ends_with(b"\n")
                    || raw[0] == b'\n'
                    || raw.windows(2).any(|pair| pair == b"\n\n")))
        {
            return Err("journal changed or ends with an incomplete record".into());
        }
        let mut observations = BTreeMap::new();
        let mut corrections = Vec::new();
        let mut sequence = 0u64;
        let mut tail = None;
        for line in raw
            .split(|byte| *byte == b'\n')
            .take_while(|line| !line.is_empty())
        {
            let entry: Entry = serde_json::from_slice(line).map_err(|_| "invalid journal entry")?;
            if entry.schema_version != SCHEMA
                || sequence.checked_add(1) != Some(entry.sequence)
                || entry.previous_sha256 != tail
                || entry.sha256 != digest(entry.sequence, &entry.previous_sha256, &entry.event)
            {
                return Err("journal chain verification failed".into());
            }
            match &entry.event {
                Event::Observation { observation } => {
                    if !valid_observation(observation) {
                        return Err("invalid journal observation".into());
                    }
                    observations.insert(entry.sequence, observation.clone());
                }
                Event::Correction {
                    observation_sequence,
                    snapshot_sha256,
                    node,
                    choice,
                    reviewer,
                    attested,
                } => {
                    let observation = observations
                        .get(observation_sequence)
                        .ok_or("correction references an unknown observation")?;
                    if !attested
                        || reviewer.trim().is_empty()
                        || node.trim().is_empty()
                        || choice.trim().is_empty()
                        || snapshot_sha256 != &observation.snapshot_sha256
                        || (node != &observation.at_node
                            && !observation.choices.iter().any(|item| &item.node == node))
                    {
                        return Err("invalid journal correction".into());
                    }
                    corrections.push((entry.sequence, entry.event.clone()));
                }
            }
            sequence = entry.sequence;
            tail = Some(entry.sha256);
        }
        file.seek(SeekFrom::End(0))
            .map_err(|error| error.to_string())?;
        Ok(Self {
            file,
            max_bytes,
            length: before,
            sequence,
            tail,
            observations,
            corrections,
        })
    }

    fn append(&mut self, event: Event) -> Result<u64, String> {
        if self
            .file
            .metadata()
            .map_err(|error| error.to_string())?
            .len()
            != self.length
        {
            return Err("journal changed while open".into());
        }
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or("journal sequence overflow")?;
        let hash = digest(sequence, &self.tail, &event);
        let entry = Entry {
            schema_version: SCHEMA.into(),
            sequence,
            previous_sha256: self.tail.clone(),
            event,
            sha256: hash.clone(),
        };
        let mut bytes = serde_json::to_vec(&entry).map_err(|error| error.to_string())?;
        bytes.push(b'\n');
        let next_length = self
            .length
            .checked_add(bytes.len() as u64)
            .ok_or("journal size overflow")?;
        if next_length > self.max_bytes {
            return Err("journal exceeds configured size limit".into());
        }
        self.file
            .write_all(&bytes)
            .map_err(|error| format!("journal write failed: {error}"))?;
        self.file
            .sync_data()
            .map_err(|error| format!("journal sync failed: {error}"))?;
        self.length = next_length;
        self.sequence = sequence;
        self.tail = Some(hash);
        match entry.event {
            Event::Observation { observation } => {
                self.observations.insert(sequence, observation);
            }
            correction @ Event::Correction { .. } => {
                self.corrections.push((sequence, correction));
            }
        }
        Ok(sequence)
    }

    pub fn review_queue(
        &self,
        engine: &lfg_core::kernel::Engine,
        source: &SourceIndex,
        output: &Path,
        max_output_bytes: u64,
    ) -> Result<Value, String> {
        if max_output_bytes == 0 {
            return Err("output size limit must be positive".into());
        }
        let reconciliation = self.reconcile(source)?;
        if reconciliation["missing"] != 0 {
            return Err("source must cover every journal observation".into());
        }
        let mut corrected = std::collections::BTreeSet::new();
        for (_, event) in &self.corrections {
            if let Event::Correction {
                observation_sequence,
                node,
                ..
            } = event
                && !corrected.insert((*observation_sequence, node.as_str()))
            {
                return Err(
                    "multiple corrections for one observation and node require resolution".into(),
                );
            }
        }
        let mut bytes = Vec::new();
        let mut nodes = BTreeMap::<String, u64>::new();
        let mut already_corrected = 0u64;
        let mut baseline_selected = 0u64;
        let mut baseline_unselected = 0u64;
        for (sequence, observation) in &self.observations {
            if observation.snapshot_sha256 != engine.snapshot_digest() {
                return Err("observation does not match the supplied snapshot".into());
            }
            let text = source
                .rows
                .get(&observation.request_id)
                .ok_or("source missing observation")?;
            let mut review_nodes = BTreeMap::new();
            for choice in &observation.choices {
                if !engine.has_choice(&choice.node, &choice.option)
                    || review_nodes
                        .insert(choice.node.as_str(), Some(choice.option.as_str()))
                        .is_some()
                {
                    return Err("observation contains invalid or duplicate node choices".into());
                }
            }
            if engine.choice_options(&observation.at_node).is_some() {
                review_nodes.entry(&observation.at_node).or_insert(None);
            }
            for (node, baseline) in review_nodes {
                if corrected.contains(&(*sequence, node)) {
                    already_corrected += 1;
                    continue;
                }
                let options = engine
                    .choice_options(node)
                    .ok_or("observation node is not a choice")?;
                let row = json!({"schemaVersion":"lfg.node-review-queue-item.v1", "status":"pending_review",
                    "requestId":observation.request_id, "text":text, "inputSha256":observation.input_sha256,
                    "sourceFileSha256":source.sha256, "snapshotSha256":observation.snapshot_sha256,
                    "snapshotRevision":observation.snapshot_revision, "journalTailSha256":self.tail,
                    "observationSequence":sequence, "node":node, "options":options,
                    "baselineState":observation.state, "baselineChoice":baseline,
                    "reviewedChoice":null, "reviewer":null, "project":null, "family":null, "split":null});
                let line = serde_json::to_vec(&row).map_err(|error| error.to_string())?;
                let next = (bytes.len() as u64)
                    .checked_add(line.len() as u64)
                    .and_then(|n| n.checked_add(1))
                    .ok_or("output size overflow")?;
                if next > max_output_bytes {
                    return Err("review queue exceeds configured size limit".into());
                }
                bytes.extend_from_slice(&line);
                bytes.push(b'\n');
                *nodes.entry(node.to_owned()).or_default() += 1;
                if baseline.is_some() {
                    baseline_selected += 1;
                } else {
                    baseline_unselected += 1;
                }
            }
        }
        if bytes.is_empty() {
            return Err("no uncorrected choice observations available for review".into());
        }
        source.verify_unchanged()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(|error| format!("cannot create review queue: {error}"))?;
        file.write_all(&bytes)
            .map_err(|error| format!("review queue write failed: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("review queue sync failed: {error}"))?;
        Ok(
            json!({"schemaVersion":"lfg.node-review-queue-report.v1", "status":"pending_review",
            "items":baseline_selected + baseline_unselected, "baselineSelected":baseline_selected,
            "baselineUnselected":baseline_unselected,"alreadyCorrected":already_corrected,
            "nodes":nodes,"sourceFileSha256":source.sha256,"journalTailSha256":self.tail,
            "sha256":format!("{:x}", sha2::Sha256::digest(&bytes))}),
        )
    }

    pub fn export_candidates(
        &self,
        engine: &lfg_core::kernel::Engine,
        output: &Path,
        max_output_bytes: u64,
        source: Option<&SourceIndex>,
    ) -> Result<Value, String> {
        if max_output_bytes == 0 {
            return Err("output size limit must be positive".into());
        }
        if let Some(source) = source {
            self.reconcile(source)?;
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut bytes = Vec::new();
        let mut exported = 0u64;
        let mut without_text = 0u64;
        for (correction_sequence, event) in &self.corrections {
            let Event::Correction {
                observation_sequence,
                snapshot_sha256,
                node,
                choice,
                reviewer,
                ..
            } = event
            else {
                unreachable!()
            };
            if !seen.insert((*observation_sequence, node)) {
                return Err(
                    "multiple corrections for one observation and node require resolution".into(),
                );
            }
            let observation = self
                .observations
                .get(observation_sequence)
                .ok_or("unknown observation")?;
            if snapshot_sha256 != engine.snapshot_digest() || !engine.has_choice(node, choice) {
                return Err("correction does not match the supplied snapshot".into());
            }
            let from_source = source.and_then(|index| index.rows.get(&observation.request_id));
            let text = observation.captured_text.as_ref().or(from_source);
            let Some(text) = text else {
                without_text += 1;
                continue;
            };
            let origin = if observation.captured_text.is_some() {
                "journal"
            } else {
                "source_jsonl"
            };
            let row = json!({"schemaVersion":"lfg.node-correction-candidate.v2", "status":"pending_review",
                "requestId":observation.request_id, "text":text, "inputSha256":observation.input_sha256,
                "textOrigin":origin,"sourceFileSha256":source.filter(|_| from_source.is_some()).map(|value| &value.sha256),
                "snapshotSha256":snapshot_sha256, "snapshotRevision":observation.snapshot_revision,
                "node":node,"proposedChoice":choice,"reviewerClaim":reviewer,
                "observationSequence":observation_sequence,"correctionSequence":correction_sequence,
                "journalTailSha256":self.tail,
                "project":null,"family":null,"split":null});
            let line = serde_json::to_vec(&row).map_err(|error| error.to_string())?;
            let next = (bytes.len() as u64)
                .checked_add(line.len() as u64)
                .and_then(|n| n.checked_add(1))
                .ok_or("output size overflow")?;
            if next > max_output_bytes {
                return Err("candidate export exceeds configured size limit".into());
            }
            bytes.extend_from_slice(&line);
            bytes.push(b'\n');
            exported += 1;
        }
        if exported == 0 {
            return Err("no corrected observations have available verified text".into());
        }
        if let Some(source) = source {
            source.verify_unchanged()?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(|error| format!("cannot create candidate export: {error}"))?;
        file.write_all(&bytes)
            .map_err(|error| format!("candidate export write failed: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("candidate export sync failed: {error}"))?;
        Ok(
            json!({"schemaVersion":"lfg.node-candidate-export.v2","candidates":exported,
            "correctedWithoutText":without_text,"sha256":format!("{:x}", sha2::Sha256::digest(&bytes)),
            "sourceFileSha256":source.map(|value| &value.sha256),
            "journalTailSha256":self.tail,
            "status":"pending_review"}),
        )
    }
}

impl Journal for JsonlJournal {
    fn observe(
        &mut self,
        id: &str,
        text: &str,
        result: &lfg_core::kernel::ResultRecord,
        capture_text: bool,
    ) -> Result<u64, String> {
        let result = serde_json::to_value(result).map_err(|error| error.to_string())?;
        let string = |key: &str| {
            result
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or("invalid kernel result")
        };
        let choices = result
            .get("hops")
            .and_then(Value::as_array)
            .ok_or("invalid kernel hops")?
            .iter()
            .filter(|hop| hop["operation"] == "choice")
            .map(|hop| {
                Ok(ChoiceObservation {
                    node: hop["node"]
                        .as_str()
                        .ok_or("invalid choice node")?
                        .to_owned(),
                    option: hop["value"]
                        .as_str()
                        .ok_or("invalid choice value")?
                        .to_owned(),
                })
            })
            .collect::<Result<Vec<_>, &str>>()?;
        let observation = Observation {
            request_id: id.to_owned(),
            input_sha256: lfg_core::digest_text(text),
            captured_text: capture_text.then(|| text.to_owned()),
            snapshot_sha256: string("snapshotSha256")?,
            snapshot_revision: string("snapshotRevision")?,
            address: string("address")?,
            state: string("state")?,
            at_node: string("atNode")?,
            choices,
        };
        if !valid_observation(&observation) {
            return Err("invalid kernel observation".into());
        }
        self.append(Event::Observation { observation })
    }

    fn correct(
        &mut self,
        engine: &lfg_core::kernel::Engine,
        observation_sequence: u64,
        node: &str,
        choice: &str,
        reviewer: &str,
    ) -> Result<u64, String> {
        let observation = self
            .observations
            .get(&observation_sequence)
            .ok_or("unknown observation")?;
        if reviewer.trim().is_empty()
            || reviewer.contains(['\n', '\r'])
            || observation.snapshot_sha256 != engine.snapshot_digest()
            || (node != observation.at_node
                && !observation.choices.iter().any(|item| item.node == node))
            || !engine.has_choice(node, choice)
        {
            return Err("correction does not match the observation or snapshot choice".into());
        }
        self.append(Event::Correction {
            observation_sequence,
            snapshot_sha256: engine.snapshot_digest().into(),
            node: node.into(),
            choice: choice.into(),
            reviewer: reviewer.into(),
            attested: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn result(engine: &lfg_core::kernel::Engine, text: &str) -> lfg_core::kernel::ResultRecord {
        engine
            .evaluate(&BTreeMap::from([("prompt".into(), text.into())]))
            .unwrap()
    }

    #[test]
    fn observations_and_attested_corrections_are_chained_without_text_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let engine = lfg_core::kernel::Engine::bundled().unwrap();
        let text = "unmatched private fixture";
        let mut journal = JsonlJournal::open(&path, 100000, true).unwrap();
        assert!(JsonlJournal::open(&path, 100000, false).is_err());
        assert_eq!(
            journal
                .observe("request-1", text, &result(&engine, text), false)
                .unwrap(),
            1
        );
        assert!(
            journal
                .correct(&engine, 1, "operation", "unknown", "reviewer")
                .is_err()
        );
        assert!(
            journal
                .correct(&engine, 1, "domain", "auth", "reviewer")
                .is_err()
        );
        assert!(
            journal
                .correct(&engine, 1, "operation", "chat", "")
                .is_err()
        );
        assert_eq!(
            journal
                .correct(&engine, 1, "operation", "chat", "reviewer")
                .unwrap(),
            2
        );
        drop(journal);
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains(text));
        let journal = JsonlJournal::open(&path, 100000, false).unwrap();
        assert_eq!(journal.status()["observations"], 1);
        assert_eq!(journal.status()["corrections"], 1);
        assert_eq!(journal.status()["capturedText"], 0);
        let shown = journal.show(&engine, 1, false, None).unwrap();
        assert_eq!(
            shown["choiceOptions"]["operation"],
            json!(["chat", "tool_call"])
        );
        assert_eq!(shown["capturedTextAvailable"], false);
        let mut other: Value =
            serde_json::from_str(include_str!("../../lfg-core/config/kernel-core.v1.json"))
                .unwrap();
        other["revision"] = json!("r2");
        let other = lfg_core::kernel::Engine::from_json(other.to_string().as_bytes()).unwrap();
        assert!(journal.show(&other, 1, false, None).is_err());
        drop(journal);
        let changed = raw.replace("reviewer", "changed-reviewer");
        std::fs::write(&path, changed).unwrap();
        assert!(JsonlJournal::open(&path, 100000, false).is_err());
    }

    #[test]
    fn text_capture_is_explicit_and_torn_or_oversized_journals_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.jsonl");
        let engine = lfg_core::kernel::Engine::bundled().unwrap();
        let mut journal = JsonlJournal::open(&path, 100000, true).unwrap();
        journal
            .observe(
                "request-1",
                "fixture text",
                &result(&engine, "fixture text"),
                true,
            )
            .unwrap();
        assert_eq!(journal.status()["capturedText"], 1);
        assert!(
            journal.show(&engine, 1, false, None).unwrap()["observation"]["capturedText"].is_null()
        );
        assert_eq!(
            journal.show(&engine, 1, true, None).unwrap()["observation"]["capturedText"],
            "fixture text"
        );
        drop(journal);
        let raw = std::fs::read(&path).unwrap();
        assert!(String::from_utf8_lossy(&raw).contains("fixture text"));
        assert!(JsonlJournal::open(&path, 1, false).is_err());
        std::fs::write(&path, &raw[..raw.len() - 1]).unwrap();
        assert!(JsonlJournal::open(&path, 100000, false).is_err());
        let mut blank = raw.clone();
        blank.push(b'\n');
        std::fs::write(&path, blank).unwrap();
        assert!(JsonlJournal::open(&path, 100000, false).is_err());
        let empty = dir.path().join("limited.jsonl");
        let mut limited = JsonlJournal::open(&empty, 1, true).unwrap();
        assert!(
            limited
                .observe("a", "text", &result(&engine, "text"), false)
                .is_err()
        );
        assert_eq!(std::fs::metadata(&empty).unwrap().len(), 0);
    }

    #[test]
    fn candidate_export_is_pending_only_bounded_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let journal_path = dir.path().join("journal.jsonl");
        let output = dir.path().join("candidates.jsonl");
        let engine = lfg_core::kernel::Engine::bundled().unwrap();
        let mut journal = JsonlJournal::open(&journal_path, 100000, true).unwrap();
        journal
            .observe(
                "example",
                "fixture text",
                &result(&engine, "fixture text"),
                true,
            )
            .unwrap();
        journal
            .correct(&engine, 1, "operation", "chat", "reviewer")
            .unwrap();
        assert!(
            journal
                .export_candidates(&engine, &output, 1, None)
                .is_err()
        );
        assert!(!output.exists());
        drop(journal);
        let mut journal = JsonlJournal::open(&journal_path, 100000, false).unwrap();
        let report = journal
            .export_candidates(&engine, &output, 100000, None)
            .unwrap();
        assert_eq!(report["candidates"], 1);
        let row: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(row["status"], "pending_review");
        assert_eq!(row["proposedChoice"], "chat");
        assert!(row["project"].is_null() && row["family"].is_null() && row["split"].is_null());
        assert!(
            journal
                .export_candidates(&engine, &output, 100000, None)
                .is_err()
        );
        journal
            .correct(&engine, 1, "operation", "tool_call", "reviewer")
            .unwrap();
        assert!(
            journal
                .export_candidates(&engine, &dir.path().join("ambiguous.jsonl"), 100000, None)
                .is_err()
        );
    }

    #[test]
    fn source_reconciliation_and_export_bind_to_exact_input_without_changing_journal() {
        let dir = tempfile::tempdir().unwrap();
        let journal_path = dir.path().join("journal.jsonl");
        let source_path = dir.path().join("source.jsonl");
        let output = dir.path().join("candidates.jsonl");
        let engine = lfg_core::kernel::Engine::bundled().unwrap();
        let mut journal = JsonlJournal::open(&journal_path, 100000, true).unwrap();
        journal
            .observe(
                "example",
                "private fixture",
                &result(&engine, "private fixture"),
                false,
            )
            .unwrap();
        journal
            .observe(
                "not-in-source",
                "another fixture",
                &result(&engine, "another fixture"),
                false,
            )
            .unwrap();
        journal
            .correct(&engine, 1, "operation", "chat", "reviewer")
            .unwrap();
        std::fs::write(
            &source_path,
            b"{\"id\":\"example\",\"text\":\"private fixture\",\"extra\":true}\n",
        )
        .unwrap();
        let source = SourceIndex::read(&source_path, 100000).unwrap();
        let summary = journal.reconcile(&source).unwrap();
        assert_eq!(summary["matched"], 1);
        assert_eq!(summary["missing"], 1);
        assert_eq!(summary["reviewed"], false);
        let hidden = journal.show(&engine, 1, false, Some(&source)).unwrap();
        assert_eq!(hidden["sourceTextAvailable"], true);
        assert!(hidden["sourceText"].is_null());
        assert_eq!(
            journal.show(&engine, 1, true, Some(&source)).unwrap()["sourceText"],
            "private fixture"
        );
        assert!(
            !serde_json::to_string(&summary)
                .unwrap()
                .contains("private fixture")
        );
        drop(journal);
        let before = std::fs::read(&journal_path).unwrap();
        let journal = JsonlJournal::open(&journal_path, 100000, false).unwrap();
        let report = journal
            .export_candidates(&engine, &output, 100000, Some(&source))
            .unwrap();
        let row: Value = serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(row["status"], "pending_review");
        assert_eq!(row["textOrigin"], "source_jsonl");
        assert_eq!(row["schemaVersion"], "lfg.node-correction-candidate.v2");
        assert_eq!(row["sourceFileSha256"], summary["sourceSha256"]);
        assert_eq!(row["text"], "private fixture");
        assert_eq!(report["candidates"], 1);
        drop(journal);
        assert_eq!(std::fs::read(&journal_path).unwrap(), before);
        let journal = JsonlJournal::open(&journal_path, 100000, false).unwrap();
        std::fs::write(
            &source_path,
            b"{\"id\":\"example\",\"text\":\"changed fixture\"}\n",
        )
        .unwrap();
        assert!(journal.reconcile(&source).is_err());
        let changed = SourceIndex::read(&source_path, 100000).unwrap();
        assert!(journal.reconcile(&changed).is_err());
        let new_output = dir.path().join("wrong.jsonl");
        assert!(
            journal
                .export_candidates(&engine, &new_output, 100000, Some(&changed))
                .is_err()
        );
        assert!(!new_output.exists());
        std::fs::write(
            &source_path,
            b"{\"id\":\"x\",\"text\":\"a\"}\n{\"id\":\"x\",\"text\":\"b\"}\n",
        )
        .unwrap();
        assert!(SourceIndex::read(&source_path, 100000).is_err());
        assert!(SourceIndex::read(&source_path, 1).is_err());
    }
}
