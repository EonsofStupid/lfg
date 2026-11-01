use lfg_core::kernel::Engine;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn snapshot() -> Value {
    json!({"schemaVersion":"lfg.kernel-snapshot.v1", "revision":"r1", "namespace":"rr1:core",
    "keySeed":7, "maxInputBytes":5000, "maxRegexBytes":1000000, "root":"gate",
    "nodes":{
        "gate":{"kind":"gate","next":"choice","rules":[
            {"id":"has-project","field":"project","pattern":"^yes$","open":true},
            {"id":"no-project","field":"project","pattern":"^no$","open":false}]},
        "choice":{"kind":"choice","thresholdPpm":600000,"ambiguityPpm":50000,
            "options":{"fix":"score","explain":"score"},"rules":[
            {"id":"fix-rule","field":"prompt","pattern":"fix","option":"fix","weightPpm":800000},
            {"id":"explain-rule","field":"prompt","pattern":"explain","option":"explain","weightPpm":800000}]},
        "score":{"kind":"score","slot":"complexity","next":"extract","rules":[
            {"id":"score-rule","field":"prompt","pattern":"fix|explain","value":2,"weightPpm":500000}]},
        "extract":{"kind":"extract","slot":"framework","next":"done","rules":[
            {"id":"framework-rule","field":"prompt","pattern":"(?P<framework>rust|go)","capture":"framework"}]},
        "done":{"kind":"compose"}
    }})
}

fn evaluate(snapshot: &Value, project: Option<&str>, prompt: &str) -> Value {
    let engine = Engine::from_json(snapshot.to_string().as_bytes()).unwrap();
    let mut fields = BTreeMap::from([("prompt".to_string(), prompt.to_string())]);
    if let Some(value) = project {
        fields.insert("project".into(), value.into());
    }
    serde_json::to_value(engine.evaluate(&fields).unwrap()).unwrap()
}

#[test]
fn five_operations_execute_and_revisions_do_not_change_path_keys() {
    let config = snapshot();
    let result = evaluate(&config, Some("yes"), "fix rust");
    assert_eq!(result["state"], "complete");
    assert_eq!(result["address"], "rr1:core/fix@r1");
    assert_eq!(result["scores"]["complexity"], 2);
    assert_eq!(result["extracts"]["framework"], "rust");
    assert_eq!(result["hops"][1]["ruleIds"], json!(["fix-rule"]));
    assert_eq!(result["hops"][1]["ruleFields"], json!(["prompt"]));
    assert_eq!(result["hops"][1]["ruleWeightsPpm"], json!([800000]));
    assert_eq!(
        result["hops"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x["operation"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["gate", "choice", "score", "extract", "compose"]
    );
    assert_eq!(result, evaluate(&config, Some("yes"), "fix rust"));
    let mut new_revision = config.clone();
    new_revision["revision"] = json!("r2");
    let changed = evaluate(&new_revision, Some("yes"), "fix rust");
    assert_eq!(changed["address"], "rr1:core/fix@r2");
    assert_eq!(result["key64"], changed["key64"]);
    assert_ne!(result["snapshotSha256"], changed["snapshotSha256"]);
}

#[test]
fn missing_evidence_gate_pruning_and_ambiguous_choice_are_explicit() {
    let config = snapshot();
    let missing = evaluate(&config, None, "fix rust");
    assert_eq!(missing["state"], "escalate");
    assert_eq!(missing["reason"], "missing_input");
    assert_eq!(evaluate(&config, Some("no"), "fix rust")["state"], "pruned");
    assert_eq!(
        evaluate(&config, Some("yes"), "unmatched rust")["reason"],
        "no_matching_rule"
    );
    let mut low_threshold = config.clone();
    low_threshold["nodes"]["choice"]["thresholdPpm"] = json!(0);
    let ambiguous = evaluate(&low_threshold, Some("yes"), "fix explain rust");
    assert_eq!(ambiguous["reason"], "ambiguous_choice");
    assert_eq!(ambiguous["address"], "rr1:core@r1");
    assert_eq!(
        evaluate(&config, Some("yes"), "fix unknown")["reason"],
        "no_matching_capture"
    );
    let mut conflict = config.clone();
    conflict["nodes"]["gate"]["rules"][1]["pattern"] = json!("^yes$");
    let result = evaluate(&conflict, Some("yes"), "fix rust");
    assert_eq!(result["reason"], "conflicting_gate_rules");
    assert_eq!(
        result["hops"][0]["ruleIds"],
        json!(["has-project", "no-project"])
    );
}

#[test]
fn optional_minimum_evidence_gates_raw_support_without_changing_legacy_snapshots() {
    let mut config = snapshot();
    config["nodes"]["choice"]["rules"][0]["weightPpm"] = json!(300000);
    let legacy = evaluate(&config, Some("yes"), "fix rust");
    assert_eq!(legacy["state"], "complete");
    assert_eq!(legacy["hops"][1]["topScorePpm"], 1000000);
    assert!(legacy["hops"][1].get("evidencePpm").is_none());

    config["nodes"]["choice"]["minEvidencePpm"] = json!(500000);
    let insufficient = evaluate(&config, Some("yes"), "fix rust");
    assert_eq!(insufficient["state"], "escalate");
    assert_eq!(insufficient["reason"], "below_minimum_evidence");
    assert_eq!(insufficient["hops"][1]["topScorePpm"], 1000000);
    assert_eq!(insufficient["hops"][1]["evidencePpm"], 300000);

    config["nodes"]["choice"]["rules"].as_array_mut().unwrap().push(json!(
        {"id":"second-fix-rule","field":"prompt","pattern":"rust","option":"fix","weightPpm":300000}
    ));
    let sufficient = evaluate(&config, Some("yes"), "fix rust");
    assert_eq!(sufficient["state"], "complete");
    assert_eq!(sufficient["hops"][1]["evidencePpm"], 510000);
    assert_eq!(sufficient, evaluate(&config, Some("yes"), "fix rust"));
    config["nodes"]["choice"]["minEvidencePpm"] = json!(510000);
    assert_eq!(
        evaluate(&config, Some("yes"), "fix rust")["state"],
        "complete"
    );
    config["nodes"]["choice"]["thresholdPpm"] = json!(900000);
    let competing = evaluate(&config, Some("yes"), "fix explain rust");
    assert_eq!(competing["reason"], "below_threshold");
    assert_eq!(competing["hops"][1]["evidencePpm"], 800000);
    config["nodes"]["choice"]["minEvidencePpm"] = json!(1000001);
    assert!(Engine::from_json(config.to_string().as_bytes()).is_err());
}

#[test]
fn rejects_bad_rules_missing_targets_cycles_and_oversize_input() {
    let mut config = snapshot();
    config["nodes"]["done"] = json!({"kind":"gate","next":"gate","rules":[
        {"id":"cycle-rule","field":"prompt","pattern":"x","open":true}]});
    assert!(Engine::from_json(config.to_string().as_bytes()).is_err());
    config = snapshot();
    config["nodes"]["gate"]["next"] = json!("missing");
    assert!(Engine::from_json(config.to_string().as_bytes()).is_err());
    config = snapshot();
    config["nodes"]["extract"]["rules"][0]["capture"] = json!("missing");
    assert!(Engine::from_json(config.to_string().as_bytes()).is_err());
    config = snapshot();
    config["maxInputBytes"] = json!(1);
    let engine = Engine::from_json(config.to_string().as_bytes()).unwrap();
    assert!(
        engine
            .evaluate(&BTreeMap::from([("prompt".into(), "long".into())]))
            .is_err()
    );
}
