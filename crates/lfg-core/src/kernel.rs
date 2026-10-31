//! Experimental deterministic LFG kernel. Rules and thresholds are snapshot data.
use std::collections::{BTreeMap, BTreeSet};

use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xxhash_rust::xxh3::xxh3_64_with_seed;

const SCALE: u64 = 1_000_000;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Snapshot {
    schema_version: String,
    revision: String,
    namespace: String,
    key_seed: u64,
    max_input_bytes: usize,
    max_regex_bytes: usize,
    root: String,
    nodes: BTreeMap<String, Node>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "lowercase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Node {
    Gate {
        rules: Vec<GateRule>,
        next: String,
    },
    Choice {
        options: BTreeMap<String, String>,
        rules: Vec<ChoiceRule>,
        threshold_ppm: u32,
        ambiguity_ppm: u32,
    },
    Score {
        slot: String,
        rules: Vec<ScoreRule>,
        next: String,
    },
    Extract {
        slot: String,
        rules: Vec<ExtractRule>,
        next: String,
    },
    Compose,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GateRule {
    id: String,
    field: String,
    pattern: String,
    open: bool,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ChoiceRule {
    id: String,
    field: String,
    pattern: String,
    option: String,
    weight_ppm: u32,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ScoreRule {
    id: String,
    field: String,
    pattern: String,
    value: i32,
    weight_ppm: u32,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExtractRule {
    id: String,
    field: String,
    pattern: String,
    capture: String,
}

struct CompiledRule {
    regex: Regex,
    pattern_sha256: String,
}

pub struct Engine {
    snapshot: Snapshot,
    snapshot_sha256: String,
    rules: BTreeMap<String, CompiledRule>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResultRecord {
    schema_version: &'static str,
    snapshot_revision: String,
    snapshot_sha256: String,
    address: String,
    key64: String,
    state: &'static str,
    reason: Option<&'static str>,
    at_node: String,
    scores: BTreeMap<String, i32>,
    extracts: BTreeMap<String, String>,
    hops: Vec<Hop>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Hop {
    node: String,
    operation: &'static str,
    value: serde_json::Value,
    rule_ids: Vec<String>,
    rule_fields: Vec<String>,
    pattern_sha256: Vec<String>,
    rule_weights_ppm: Vec<Option<u32>>,
    top_score_ppm: Option<u32>,
    margin_ppm: Option<u32>,
}

fn valid_segment(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
}

impl Engine {
    pub fn snapshot_digest(&self) -> &str {
        &self.snapshot_sha256
    }

    pub fn has_choice(&self, node: &str, option: &str) -> bool {
        matches!(self.snapshot.nodes.get(node), Some(Node::Choice { options, .. }) if options.contains_key(option))
    }

    pub fn choice_options(&self, node: &str) -> Option<Vec<String>> {
        match self.snapshot.nodes.get(node) {
            Some(Node::Choice { options, .. }) => Some(options.keys().cloned().collect()),
            _ => None,
        }
    }

    pub fn bundled() -> Result<Self, String> {
        Self::from_json(include_bytes!("../config/kernel-core.v1.json"))
    }

    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let snapshot: Snapshot = serde_json::from_slice(raw).map_err(|error| error.to_string())?;
        if snapshot.schema_version != "lfg.kernel-snapshot.v1"
            || !valid_segment(&snapshot.revision)
            || !snapshot.namespace.starts_with("rr1:")
            || !valid_segment(&snapshot.namespace[4..])
            || snapshot.max_input_bytes == 0
            || snapshot.max_regex_bytes == 0
            || snapshot.nodes.is_empty()
            || !snapshot.nodes.contains_key(&snapshot.root)
        {
            return Err("invalid kernel snapshot header".into());
        }
        let mut rules = BTreeMap::new();
        let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (id, node) in &snapshot.nodes {
            if !valid_segment(id) {
                return Err("invalid node ID".into());
            }
            let mut compile = |rule_id: &str,
                               field: &str,
                               pattern: &str,
                               capture: Option<&str>|
             -> Result<(), String> {
                if !valid_segment(rule_id)
                    || !valid_segment(field)
                    || pattern.is_empty()
                    || rules.contains_key(rule_id)
                {
                    return Err("invalid or duplicate rule ID, field, or pattern".into());
                }
                let regex = RegexBuilder::new(pattern)
                    .size_limit(snapshot.max_regex_bytes)
                    .build()
                    .map_err(|e| e.to_string())?;
                if capture
                    .is_some_and(|name| !regex.capture_names().flatten().any(|value| value == name))
                {
                    return Err("extract rule capture is absent from pattern".into());
                }
                rules.insert(
                    rule_id.to_string(),
                    CompiledRule {
                        regex,
                        pattern_sha256: format!("{:x}", Sha256::digest(pattern.as_bytes())),
                    },
                );
                Ok(())
            };
            let targets = match node {
                Node::Gate { rules, next } => {
                    if rules.is_empty() {
                        return Err("gate requires rules".into());
                    }
                    for r in rules {
                        compile(&r.id, &r.field, &r.pattern, None)?;
                    }
                    vec![next.clone()]
                }
                Node::Choice {
                    options,
                    rules,
                    threshold_ppm,
                    ambiguity_ppm,
                } => {
                    if options.is_empty()
                        || rules.is_empty()
                        || u64::from(*threshold_ppm) > SCALE
                        || u64::from(*ambiguity_ppm) > SCALE
                        || options.keys().any(|option| !valid_segment(option))
                    {
                        return Err("invalid choice configuration".into());
                    }
                    for r in rules {
                        if !options.contains_key(&r.option)
                            || r.weight_ppm == 0
                            || u64::from(r.weight_ppm) > SCALE
                        {
                            return Err("invalid choice rule".into());
                        }
                        compile(&r.id, &r.field, &r.pattern, None)?;
                    }
                    options.values().cloned().collect()
                }
                Node::Score { slot, rules, next } => {
                    if !valid_segment(slot) || rules.is_empty() {
                        return Err("invalid score node".into());
                    }
                    for r in rules {
                        if r.weight_ppm == 0 || u64::from(r.weight_ppm) > SCALE {
                            return Err("invalid score weight".into());
                        }
                        compile(&r.id, &r.field, &r.pattern, None)?;
                    }
                    vec![next.clone()]
                }
                Node::Extract { slot, rules, next } => {
                    if !valid_segment(slot) || rules.is_empty() {
                        return Err("invalid extract node".into());
                    }
                    for r in rules {
                        if !valid_segment(&r.capture) {
                            return Err("invalid capture name".into());
                        }
                        compile(&r.id, &r.field, &r.pattern, Some(&r.capture))?;
                    }
                    vec![next.clone()]
                }
                Node::Compose => vec![],
            };
            if targets
                .iter()
                .any(|target| !snapshot.nodes.contains_key(target))
            {
                return Err("node target is missing".into());
            }
            edges.insert(id.clone(), targets);
        }
        // Kahn's algorithm avoids recursion even for a large untrusted graph.
        let mut incoming: BTreeMap<&str, usize> =
            edges.keys().map(|name| (name.as_str(), 0)).collect();
        for targets in edges.values() {
            for target in targets {
                *incoming.get_mut(target.as_str()).expect("validated target") += 1;
            }
        }
        let mut ready: BTreeSet<&str> = incoming
            .iter()
            .filter_map(|(node, count)| (*count == 0).then_some(*node))
            .collect();
        let mut visited = 0usize;
        while let Some(node) = ready.pop_first() {
            visited += 1;
            for target in &edges[node] {
                let count = incoming.get_mut(target.as_str()).expect("validated target");
                *count -= 1;
                if *count == 0 {
                    ready.insert(target);
                }
            }
        }
        if visited != edges.len() {
            return Err("kernel graph contains a cycle".into());
        }
        Ok(Self {
            snapshot,
            snapshot_sha256: format!("{:x}", Sha256::digest(raw)),
            rules,
        })
    }

    pub fn evaluate(&self, fields: &BTreeMap<String, String>) -> Result<ResultRecord, String> {
        let bytes = fields
            .iter()
            .try_fold(0usize, |total, (key, value)| {
                total.checked_add(key.len())?.checked_add(value.len())
            })
            .ok_or("input byte count overflow")?;
        if bytes > self.snapshot.max_input_bytes || fields.keys().any(|key| !valid_segment(key)) {
            return Err("invalid or oversized kernel input".into());
        }
        let mut node = self.snapshot.root.as_str();
        let mut path = self.snapshot.namespace.clone();
        let mut scores = BTreeMap::new();
        let mut extracts = BTreeMap::new();
        let mut hops = Vec::new();
        let (state, reason) = loop {
            let mut ids = Vec::new();
            let mut rule_fields = Vec::new();
            let mut patterns = Vec::new();
            let mut weights = Vec::new();
            let mut record = |id: &str, field: &str, weight: Option<u32>| {
                ids.push(id.to_string());
                rule_fields.push(field.to_string());
                patterns.push(self.rules[id].pattern_sha256.clone());
                weights.push(weight);
            };
            let missing = |field: &str| !fields.contains_key(field);
            match &self.snapshot.nodes[node] {
                Node::Gate { rules, next } => {
                    if rules.iter().any(|r| missing(&r.field)) {
                        break ("escalate", Some("missing_input"));
                    }
                    let mut value = None;
                    let mut conflict = false;
                    for r in rules {
                        if self.rules[&r.id].regex.is_match(&fields[&r.field]) {
                            record(&r.id, &r.field, None);
                            if value.is_some_and(|old| old != r.open) {
                                conflict = true;
                            } else {
                                value = Some(r.open);
                            }
                        }
                    }
                    if conflict {
                        hops.push(Hop {
                            node: node.into(),
                            operation: "gate",
                            value: serde_json::Value::Null,
                            rule_ids: ids,
                            rule_fields,
                            pattern_sha256: patterns,
                            rule_weights_ppm: weights,
                            top_score_ppm: None,
                            margin_ppm: None,
                        });
                        break ("escalate", Some("conflicting_gate_rules"));
                    }
                    let Some(value) = value else {
                        break ("escalate", Some("no_matching_rule"));
                    };
                    hops.push(Hop {
                        node: node.into(),
                        operation: "gate",
                        value: value.into(),
                        rule_ids: ids,
                        rule_fields,
                        pattern_sha256: patterns,
                        rule_weights_ppm: weights,
                        top_score_ppm: None,
                        margin_ppm: None,
                    });
                    if !value {
                        break ("pruned", Some("gate_closed"));
                    }
                    node = next;
                }
                Node::Choice {
                    options,
                    rules,
                    threshold_ppm,
                    ambiguity_ppm,
                } => {
                    if rules.iter().any(|r| missing(&r.field)) {
                        break ("escalate", Some("missing_input"));
                    }
                    let mut totals: BTreeMap<&str, u64> =
                        options.keys().map(|key| (key.as_str(), 0)).collect();
                    for r in rules {
                        if self.rules[&r.id].regex.is_match(&fields[&r.field]) {
                            record(&r.id, &r.field, Some(r.weight_ppm));
                            let total =
                                totals.get_mut(r.option.as_str()).expect("validated option");
                            *total = SCALE
                                - ((SCALE - *total) * (SCALE - u64::from(r.weight_ppm))
                                    + SCALE / 2)
                                    / SCALE;
                        }
                    }
                    let sum: u64 = totals.values().sum();
                    if sum == 0 {
                        break ("escalate", Some("no_matching_rule"));
                    }
                    let mut ranked: Vec<_> = totals.into_iter().collect();
                    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                    let top = (ranked[0].1 * SCALE / sum) as u32;
                    let second = ranked
                        .get(1)
                        .map_or(0, |item| (item.1 * SCALE / sum) as u32);
                    let margin = top - second;
                    hops.push(Hop {
                        node: node.into(),
                        operation: "choice",
                        value: ranked[0].0.into(),
                        rule_ids: ids,
                        rule_fields,
                        pattern_sha256: patterns,
                        rule_weights_ppm: weights,
                        top_score_ppm: Some(top),
                        margin_ppm: Some(margin),
                    });
                    if top < *threshold_ppm {
                        break ("escalate", Some("below_threshold"));
                    }
                    if margin <= *ambiguity_ppm {
                        break ("escalate", Some("ambiguous_choice"));
                    }
                    path.push('/');
                    path.push_str(ranked[0].0);
                    node = &options[ranked[0].0];
                }
                Node::Score { slot, rules, next } => {
                    if rules.iter().any(|r| missing(&r.field)) {
                        break ("escalate", Some("missing_input"));
                    }
                    let mut total = 0i128;
                    let mut weight = 0i128;
                    for r in rules {
                        if self.rules[&r.id].regex.is_match(&fields[&r.field]) {
                            record(&r.id, &r.field, Some(r.weight_ppm));
                            total += i128::from(r.value) * i128::from(r.weight_ppm);
                            weight += i128::from(r.weight_ppm);
                        }
                    }
                    if weight == 0 {
                        break ("escalate", Some("no_matching_rule"));
                    }
                    let value = (total / weight) as i32;
                    if scores.insert(slot.clone(), value).is_some() {
                        return Err("duplicate score slot on execution path".into());
                    }
                    hops.push(Hop {
                        node: node.into(),
                        operation: "score",
                        value: value.into(),
                        rule_ids: ids,
                        rule_fields,
                        pattern_sha256: patterns,
                        rule_weights_ppm: weights,
                        top_score_ppm: None,
                        margin_ppm: None,
                    });
                    node = next;
                }
                Node::Extract { slot, rules, next } => {
                    if rules.iter().any(|r| missing(&r.field)) {
                        break ("escalate", Some("missing_input"));
                    }
                    let mut value = None;
                    for r in rules {
                        if let Some(captures) = self.rules[&r.id].regex.captures(&fields[&r.field])
                        {
                            let Some(matched) = captures.name(&r.capture) else {
                                break;
                            };
                            record(&r.id, &r.field, None);
                            value = Some(matched.as_str().to_string());
                            break;
                        }
                    }
                    let Some(value) = value else {
                        break ("escalate", Some("no_matching_capture"));
                    };
                    if extracts.insert(slot.clone(), value.clone()).is_some() {
                        return Err("duplicate extract slot on execution path".into());
                    }
                    hops.push(Hop {
                        node: node.into(),
                        operation: "extract",
                        value: value.into(),
                        rule_ids: ids,
                        rule_fields,
                        pattern_sha256: patterns,
                        rule_weights_ppm: weights,
                        top_score_ppm: None,
                        margin_ppm: None,
                    });
                    node = next;
                }
                Node::Compose => {
                    hops.push(Hop {
                        node: node.into(),
                        operation: "compose",
                        value: serde_json::Value::Null,
                        rule_ids: ids,
                        rule_fields,
                        pattern_sha256: patterns,
                        rule_weights_ppm: weights,
                        top_score_ppm: None,
                        margin_ppm: None,
                    });
                    break ("complete", None);
                }
            }
        };
        let key = xxh3_64_with_seed(path.as_bytes(), self.snapshot.key_seed);
        Ok(ResultRecord {
            schema_version: "lfg.kernel-result.v1",
            snapshot_revision: self.snapshot.revision.clone(),
            snapshot_sha256: self.snapshot_sha256.clone(),
            address: format!("{}@{}", path, self.snapshot.revision),
            key64: format!("{key:016x}"),
            state,
            reason,
            at_node: node.into(),
            scores,
            extracts,
            hops,
        })
    }
}
