use super::models::effort_for;
use super::*;

fn rules() -> PreflightRuleConfig {
    PreflightRuleConfig::from_json_str(include_str!("../../config/preflight.v1.json")).unwrap()
}

fn request(text: &str, declared: &[&str]) -> AssessRequest {
    AssessRequest {
        session_id: "test".into(),
        revision: 1,
        text: text.into(),
        declared_subscriptions: declared.iter().map(|s| s.to_string()).collect(),
    }
}

fn model(subscription: &str, name: &str, description: &str, priority: u32) -> KnownModel {
    KnownModel {
        subscription_id: subscription.into(),
        model: name.into(),
        label: name.into(),
        description: description.into(),
        priority: Some(priority),
        efforts: vec!["low".into(), "medium".into(), "high".into(), "xhigh".into()],
        origin: "provider-list".into(),
        available: true,
        ..KnownModel::default()
    }
}

fn used(mut known: KnownModel, bracket: &str, prompts: u64, effort: &str) -> KnownModel {
    known.usage.push(BracketUsage {
        bracket: bracket.into(),
        prompts,
        weighted: prompts as f64,
        efforts: BTreeMap::from([(effort.to_string(), prompts)]),
    });
    known
}

fn knowledge(models: Vec<KnownModel>) -> Knowledge {
    Knowledge {
        state: "ready".into(),
        subscriptions: vec![ObservedSubscription {
            id: "codex".into(),
            listed_at: Some("2026-11-01".into()),
            ..ObservedSubscription::default()
        }],
        models,
        ..Knowledge::default()
    }
}

const MULTI: &str =
    "Fix the auth token refresh bug in the Rust server, add regression tests, and update the docs";

fn configured_fixture_models() -> AssessmentConfig {
    let mut config = AssessmentConfig::default_alpha();
    config.published_models = vec![
        PublishedModel {
            subscription_id: "claude".into(),
            model: "fixture-frontier".into(),
            label: "Fixture frontier".into(),
            description: "frontier".into(),
            efforts: vec!["low".into(), "high".into()],
            source: "test".into(),
            observed_at: "fixture".into(),
        },
        PublishedModel {
            subscription_id: "claude".into(),
            model: "fixture-fast".into(),
            label: "Fixture fast".into(),
            description: "fast".into(),
            efforts: vec!["low".into(), "high".into()],
            source: "test".into(),
            observed_at: "fixture".into(),
        },
    ];
    config
}

#[test]
fn bundled_configuration_is_valid() {
    let config = AssessmentConfig::default_alpha();
    assert_eq!(config.brackets[0].min_score, 0.0);
    assert_eq!(config.digest().len(), 64);
}

#[test]
fn words_match_whole_words_with_configured_endings() {
    let suffixes = AssessmentConfig::default_alpha().word_suffixes;
    for (text, word) in [
        ("run the tests now", "test"),
        ("testing it", "test"),
        ("updating the docs", "update"),
        ("updated docs", "update"),
        ("set up the server", "set up"),
    ] {
        assert!(contains_word(text, word, &suffixes), "{text} / {word}");
    }
    for (text, word) in [
        ("the latest news", "test"),
        ("a prefix here", "fix"),
        ("tokenizer settings", "token"),
        ("contest", "test"),
    ] {
        assert!(!contains_word(text, word, &suffixes), "{text} / {word}");
    }
}

#[test]
fn all_four_results_come_from_one_formula() {
    let config = configured_fixture_models();
    let result = assess(
        &config,
        &rules(),
        &request(MULTI, &["claude"]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    assert_eq!(result.intent.primary, "build.software");
    assert_eq!(result.intent.all.len(), 2);
    assert!(result.intent.all[0].matched.contains(&"add".to_string()));
    assert_eq!(result.prompt.tasks.len(), 3);
    assert_eq!(result.complexity.bracket, "complex");
    assert_eq!(result.agents.total, 3);
    let points: f64 = result.complexity.terms.iter().map(|t| t.points).sum();
    assert!((points - result.complexity.score).abs() < 0.03);
    assert_eq!(
        result
            .complexity
            .ladder
            .iter()
            .filter(|s| s.current)
            .count(),
        1
    );
    assert_eq!(result.models.role, "frontier");
    let suggested = result.models.suggested.unwrap();
    assert_eq!(suggested.model, "fixture-frontier");
    assert_eq!(suggested.origin, "published");
    assert_eq!(suggested.effort.as_deref(), Some("high"));
    assert!(result.rules.contains(&"bracket:complex".to_string()));
    assert!(result.rules.iter().all(|rule| !rule.contains("refresh")));
}

#[test]
fn simple_requests_stay_small() {
    let config = configured_fixture_models();
    for text in ["hey thanks", "What is a closure in Rust?"] {
        let result = assess(
            &config,
            &rules(),
            &request(text, &["claude"]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
        );
        assert_eq!(result.complexity.bracket, "trivial", "{text}");
        assert_eq!(result.agents.total, 1);
        assert_eq!(result.intent.primary, "respond.general");
        assert!(result.intent.fallback);
        let suggested = result.models.suggested.unwrap();
        assert_eq!(suggested.model, "fixture-fast");
        assert_eq!(suggested.effort.as_deref(), Some("low"));
    }
}

#[test]
fn intent_words_extend_the_shared_rules() {
    let config = AssessmentConfig::default_alpha();
    let result = assess(
        &config,
        &rules(),
        &request("Refactor the payment module and deploy to production", &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    assert_eq!(result.intent.primary, "build.software");
    assert!(result.models.suggested.is_none());
    assert_eq!(
        serde_json::to_value(round2(-0.0)).unwrap().to_string(),
        "0.0"
    );
}

#[test]
fn roles_come_from_provider_descriptions_and_siblings() {
    let config = AssessmentConfig::default_alpha();
    let models = known_models(
        &config,
        &knowledge(vec![
            model(
                "codex",
                "gpt-6-luna",
                "Fast and affordable model for easier tasks.",
                5,
            ),
            model(
                "codex",
                "gpt-6-astra",
                "Frontier intelligence for the most demanding work.",
                2,
            ),
            model(
                "codex",
                "gpt-5.6-terra",
                "Older balanced model for straightforward work.",
                8,
            ),
            model("codex", "gpt-6.1-terra", "GPT-6.1 Terra Codex model.", 3),
            model(
                "codex",
                "gpt-daybreak-blue-latest",
                "Latest frontier agentic coding model for broad defensive cybersecurity work.",
                11,
            ),
            model("codex", "gpt-mystery", "A new model.", 12),
        ]),
    );
    let role = |name: &str| {
        models
            .iter()
            .find(|m| m.model == name)
            .and_then(|m| m.role.clone())
    };
    assert_eq!(role("gpt-6-luna").as_deref(), Some("fast"));
    assert_eq!(role("gpt-6-astra").as_deref(), Some("frontier"));
    assert_eq!(role("gpt-6.1-terra").as_deref(), Some("standard"));
    assert_eq!(
        role("gpt-daybreak-blue-latest").as_deref(),
        Some("specialist")
    );
    assert_eq!(role("gpt-mystery"), None);
}

#[test]
fn suggestions_blend_fit_with_the_users_habits() {
    let config = AssessmentConfig::default_alpha();
    let codex = |models: Vec<KnownModel>| {
        assess(
            &config,
            &rules(),
            &request("hey thanks", &["codex"]),
            &knowledge(models),
            &BTreeMap::new(),
        )
        .models
    };
    // A fast model still wins trivial work over a heavily used frontier model.
    let result = codex(vec![
        used(
            model(
                "codex",
                "gpt-6-astra",
                "Frontier intelligence for the most demanding work.",
                2,
            ),
            "trivial",
            100,
            "xhigh",
        ),
        model(
            "codex",
            "gpt-6-luna",
            "Fast and affordable model for easier tasks.",
            5,
        ),
    ]);
    assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-6-luna");
    assert_eq!(result.usual_model.as_deref(), Some("gpt-6-astra"));
    assert!(result.plain.contains("You usually use gpt-6-astra"));

    // Between two models of the same role, the one the user actually uses wins.
    let result = codex(vec![
        model(
            "codex",
            "gpt-6-luna",
            "Fast and affordable model for easier tasks.",
            5,
        ),
        used(
            model("codex", "gpt-7-luna", "Fast model for easier tasks.", 9),
            "trivial",
            5,
            "low",
        ),
    ]);
    assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-7-luna");

    // Without usage, the provider's own ordering breaks the tie.
    let result = codex(vec![
        model("codex", "gpt-7-luna", "Fast model for easier tasks.", 9),
        model(
            "codex",
            "gpt-6-luna",
            "Fast and affordable model for easier tasks.",
            5,
        ),
    ]);
    assert_eq!(result.suggested.as_ref().unwrap().model, "gpt-6-luna");
}

#[test]
fn retired_and_unusable_models_are_never_suggested() {
    let config = AssessmentConfig::default_alpha();
    let mut retired = used(
        model("codex", "gpt-old-luna", "Fast model.", 1),
        "trivial",
        500,
        "low",
    );
    retired.available = false;
    let result = assess(
        &config,
        &rules(),
        &request("hey thanks", &["codex"]),
        &knowledge(vec![
            retired,
            model("codex", "gpt-6-luna", "Fast model.", 5),
            model("other", "other-fast", "Fast model.", 1),
        ]),
        &BTreeMap::new(),
    );
    assert_eq!(result.models.suggested.unwrap().model, "gpt-6-luna");
    assert!(
        result
            .models
            .alternatives
            .iter()
            .all(|m| m.subscription_id == "codex")
    );
}

#[test]
fn subscription_status_reflects_evidence() {
    let config = AssessmentConfig::default_alpha();
    let observed = Knowledge {
        subscriptions: vec![
            ObservedSubscription {
                id: "claude".into(),
                plan: Some("Claude Max 20x".into()),
                ..ObservedSubscription::default()
            },
            ObservedSubscription {
                id: "stale".into(),
                last_used: Some("2020-01-01".into()),
                days_since_used: Some(2000),
                ..ObservedSubscription::default()
            },
        ],
        ..Knowledge::default()
    };
    let checks = BTreeMap::from([(
        "codex".to_string(),
        SubscriptionCheck {
            active: false,
            detail: "Codex is not signed in".into(),
        },
    )]);
    let result = assess(
        &config,
        &rules(),
        &request("hey", &["mine"]),
        &observed,
        &checks,
    );
    let status = |id: &str| {
        result
            .subscriptions
            .iter()
            .find(|s| s.id == id)
            .map(|s| (s.status.as_str(), s.usable))
            .unwrap()
    };
    assert_eq!(status("claude"), ("recognized", false));
    assert_eq!(status("codex"), ("signed-out", false));
    assert_eq!(status("mine"), ("declared", true));
    assert_eq!(status("stale"), ("not-found", false));
}

#[test]
fn remembered_use_does_not_enable_an_undeclared_subscription() {
    let result = assess(
        &AssessmentConfig::default_alpha(),
        &rules(),
        &request("hey", &[]),
        &knowledge(vec![model("codex", "listed-model", "Fast model.", 1)]),
        &BTreeMap::new(),
    );
    assert!(result.models.suggested.is_none());
    let codex = result
        .subscriptions
        .iter()
        .find(|item| item.id == "codex")
        .unwrap();
    assert_eq!(codex.status, "recognized");
    assert!(!codex.usable);
}

#[test]
fn effort_uses_the_nearest_supported_level() {
    let config = AssessmentConfig::default_alpha();
    let mut known = model("codex", "m", "Fast model.", 1);
    known.efforts = vec!["low".into(), "high".into()];
    assert_eq!(
        effort_for(&config, &known, "medium").as_deref(),
        Some("low")
    );
    assert_eq!(
        effort_for(&config, &known, "xhigh").as_deref(),
        Some("high")
    );
    known.efforts = vec!["high".into()];
    assert_eq!(effort_for(&config, &known, "low").as_deref(), Some("high"));
    known.efforts = vec![];
    assert_eq!(effort_for(&config, &known, "low"), None);
}

#[test]
fn invalid_configuration_is_rejected() {
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../config/assessment.v1.json")).unwrap();
    value["brackets"][1]["minScore"] = serde_json::json!(0.0);
    assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../config/assessment.v1.json")).unwrap();
    value["knowledgeScan"]["maxBytesPerRefresh"] = serde_json::json!(1);
    assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../config/assessment.v1.json")).unwrap();
    value["maxScoredBytes"] = serde_json::json!(0);
    assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
}

#[test]
fn scoring_limit_is_configurable_and_reported() {
    let mut config = AssessmentConfig::default_alpha();
    config.max_scored_bytes = 4;
    let result = assess(
        &config,
        &rules(),
        &request("fix the system", &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    assert!(result.prompt.scoring_truncated);
}

fn bound(config: &mut AssessmentConfig) {
    config.model_score = Some(ModelScoreBinding {
        id: "lfg-dev-text-test".into(),
        manifest_sha256: "a".repeat(64),
        level_brackets: ["trivial", "simple", "moderate", "complex", "major"]
            .map(String::from)
            .to_vec(),
        level_cut_points: vec![0.5, 1.5, 2.5, 3.5],
    });
}

fn output(probabilities: &[f64]) -> ModelOutput {
    ModelOutput {
        probabilities: probabilities.to_vec(),
        elapsed_micros: 1234,
        truncated: false,
    }
}

#[test]
fn optional_sections_leave_the_pinned_baseline_digest_unchanged() {
    let config = AssessmentConfig::default_alpha();
    assert_eq!(
        config.digest(),
        "8f89f5f4a3e7815a5ebb2fab0994c31c169cf217a82efef6cb54b02ae868ab19"
    );
    let value = serde_json::to_value(&config).unwrap();
    assert!(value.get("modelScore").is_none() && value.get("structuredInput").is_none());
    let mut bound_config = config.clone();
    bound(&mut bound_config);
    assert_ne!(bound_config.digest(), config.digest());
}

#[test]
fn structured_input_assesses_configured_fields_only() {
    let mut config = AssessmentConfig::default_alpha();
    config.structured_input = Some(StructuredInput {
        json_fields: vec!["title".into(), "goal".into()],
        separator: "\n".into(),
    });
    let task = serde_json::json!({"scope": "s", "task_id": "t-1", "title": "Fix login",
        "goal": "Make logout clear the session", "task_sha256": "f".repeat(64), "known_at_cursor": 7})
    .to_string();
    assert_eq!(
        scored_text(&config, &task),
        "Fix login\nMake logout clear the session"
    );
    assert_eq!(scored_text(&config, "plain words"), "plain words");
    assert_eq!(scored_text(&config, "{not json"), "{not json");
    assert_eq!(scored_text(&config, r#"{"other":"x"}"#), r#"{"other":"x"}"#);
    let structured = assess(
        &config,
        &rules(),
        &request(&task, &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    let plain = assess(
        &config,
        &rules(),
        &request("Fix login\nMake logout clear the session", &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    assert_eq!(structured.complexity.score, plain.complexity.score);
    assert!(structured.rules.contains(&"input:structured".to_string()));
    assert!(!plain.rules.contains(&"input:structured".to_string()));
    let mut value = serde_json::to_value(&config).unwrap();
    value["structuredInput"]["jsonFields"] = serde_json::json!(["goal", "goal"]);
    assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
}

#[test]
fn model_binding_is_validated() {
    let mut config = AssessmentConfig::default_alpha();
    bound(&mut config);
    let good = serde_json::to_value(&config).unwrap();
    assert!(AssessmentConfig::from_json_str(&good.to_string()).is_ok());
    for (field, bad) in [
        ("manifestSha256", serde_json::json!("A".repeat(64))),
        ("levelBrackets", serde_json::json!(["trivial", "nope"])),
        ("levelCutPoints", serde_json::json!([0.5, 1.5, 2.5])),
        ("levelCutPoints", serde_json::json!([0.5, 2.5, 1.5, 3.5])),
        ("id", serde_json::json!("has space")),
    ] {
        let mut value = good.clone();
        value["modelScore"][field] = bad;
        assert!(
            AssessmentConfig::from_json_str(&value.to_string()).is_err(),
            "{field} accepted"
        );
    }
    let mut value = good.clone();
    value["modelScore"]["unexpected"] = serde_json::json!(1);
    assert!(AssessmentConfig::from_json_str(&value.to_string()).is_err());
}

#[test]
fn model_level_chooses_the_bracket_and_what_follows() {
    let mut config = AssessmentConfig::default_alpha();
    bound(&mut config);
    let text = "fix typo";
    let rules_only = assess(
        &AssessmentConfig::default_alpha(),
        &rules(),
        &request(text, &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
    );
    let result = assess_with_model(
        &config,
        &rules(),
        &request(text, &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
        Some(&output(&[0.0, 0.0, 0.1, 0.8, 0.1])),
    );
    let complex = config.brackets.iter().find(|b| b.id == "complex").unwrap();
    let score = result.model_score.as_ref().unwrap();
    assert_eq!(score.status, "available");
    assert_eq!(score.expected_level, Some(3.0));
    assert_eq!(score.level, Some(3));
    assert_eq!(score.bracket.as_deref(), Some("complex"));
    assert_eq!(score.rules_bracket, rules_only.complexity.bracket);
    assert_eq!(result.complexity.bracket, "complex");
    assert_eq!(result.complexity.score, rules_only.complexity.score);
    assert_eq!(result.models.role, complex.role);
    assert_eq!(result.models.effort, complex.effort);
    assert!(result.agents.total >= complex.min_agents);
    assert!(result.rules.contains(&"bracket-source:model".to_string()));
    assert!(result.rules.contains(&"bracket:complex".to_string()));
    assert!(result.confidence.is_none());
    // Cut points are inclusive at the threshold.
    let edge = assess_with_model(
        &config,
        &rules(),
        &request(text, &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
        Some(&output(&[0.5, 0.5, 0.0, 0.0, 0.0])),
    );
    assert_eq!(edge.complexity.bracket, "simple");
}

#[test]
fn missing_or_invalid_model_output_is_reported_not_hidden() {
    let mut config = AssessmentConfig::default_alpha();
    bound(&mut config);
    let missing = assess_with_model(
        &config,
        &rules(),
        &request("fix typo", &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
        None,
    );
    assert_eq!(missing.model_score.as_ref().unwrap().status, "unavailable");
    assert!(
        missing
            .rules
            .contains(&"score-model:unavailable".to_string())
    );
    assert!(!missing.rules.contains(&"bracket-source:model".to_string()));
    for bad in [
        vec![0.5, 0.5],
        vec![0.2, 0.2, 0.2, 0.2, 0.1],
        vec![f64::NAN, 0.0, 0.0, 0.0, 1.0],
        vec![-0.5, 0.5, 0.5, 0.25, 0.25],
    ] {
        let result = assess_with_model(
            &config,
            &rules(),
            &request("fix typo", &[]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
            Some(&output(&bad)),
        );
        let score = result.model_score.unwrap();
        assert_eq!(score.status, "invalid_output");
        assert_eq!(result.complexity.bracket, score.rules_bracket);
    }
    let unbound = assess_with_model(
        &AssessmentConfig::default_alpha(),
        &rules(),
        &request("fix typo", &[]),
        &Knowledge::disabled(),
        &BTreeMap::new(),
        Some(&output(&[0.0, 0.0, 0.0, 0.0, 1.0])),
    );
    assert!(unbound.model_score.is_none());
    assert!(
        serde_json::to_value(&unbound)
            .unwrap()
            .get("modelScore")
            .is_none()
    );
}
