//! Golden outputs for the assessment. Reorganizing the classifier must not change
//! what LFG reports: every field of every assessment below is pinned by digest.
//!
//! Each prompt runs three ways: rules only with no knowledge; rules with learned
//! model knowledge and subscriptions; and the trained-model path, with valid,
//! invalid and missing model output, structured input and the same knowledge.
//!
//! `LFG_GOLDEN_WRITE=1` rewrites the fixture from the current code. Set
//! `LFG_GOLDEN_PROMPTS` to a JSONL file of `{"text": ...}` rows and
//! `LFG_GOLDEN_OUT` to a path to write digests for a private prompt set, so two
//! builds can be compared without the prompts leaving the machine.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lfg_core::assessment::{
    AssessRequest, AssessmentConfig, BracketUsage, Knowledge, KnownModel, ModelOutput,
    ModelScoreBinding, ObservedSubscription, StructuredInput, SubscriptionCheck, assess_with_model,
};
use lfg_core::{PreflightRuleConfig, digest_value};

const PROMPTS: &[&str] = &[
    "hello",
    "",
    "   ",
    "what time is it in tokyo",
    "why does the build fail on windows after the last merge",
    "fix the login timeout in the auth service so the tests pass",
    "refactor the payment module, add tests, and update the docs",
    "Research the best vector database for 10M embeddings and compare pricing",
    "draw a logo of a fox wearing a space helmet",
    "summarize this thread in three bullet points",
    "rewrite the whole backend in rust, migrate the database, then deploy to production and verify every endpoint",
    "delete all user records older than 30 days from the production database",
    "rotate the API keys and update the secrets in the deployment pipeline",
    "run the test suite and tell me what failed",
    "spin up three agents in parallel: one to audit security, one to profile performance, one to write docs",
    "can you explain how the kv cache works in transformers?",
    "update the README",
    "Updating the CI workflow to pin actions; also add a lint step and fix the failing type check",
    "investigate why memory usage grows over time in the indexer and propose a fix",
    "translate this paragraph into french",
    "build a dashboard that shows latency p50 and p95 per endpoint, with alerts when p95 exceeds 500 ms",
    "verify the release build works on arm64 linux and macos",
    "make it faster",
    "ok",
    "thanks, that worked",
    "WHY IS THIS STILL BROKEN",
    "add a --dry-run flag to the cli and document it",
    "evaluate whether the classifier beats the baseline on the held-out set",
    "migrate every service across the whole organization to the new auth provider and update all clients",
    "write a short poem about rain",
    "{\"title\":\"Fix flaky test\",\"description\":\"The scheduler test fails one run in ten; find the race and fix it.\"}",
    "{\"title\":\"\",\"description\":\"\"}",
    "{not json at all",
    "Ünïcödé prompt: réparer le bug d'authentification 🙂 and add tests",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "set up the staging environment, seed the database, run the migrations and smoke test the api",
    "is it safe to store passwords with md5",
    "plan the next sprint: list the open bugs, estimate them, and assign owners",
    "look into the failing deploy",
    "generate an image and then write alt text for it",
];

fn rules() -> PreflightRuleConfig {
    PreflightRuleConfig::from_json_str(include_str!("../config/preflight.v1.json")).unwrap()
}

fn usage(bracket: &str, prompts: u64, weighted: f64, effort: &str) -> BracketUsage {
    BracketUsage {
        bracket: bracket.into(),
        prompts,
        weighted,
        efforts: BTreeMap::from([(effort.to_string(), prompts)]),
    }
}

fn model(
    subscription: &str,
    name: &str,
    description: &str,
    priority: Option<u32>,
    usage: Vec<BracketUsage>,
) -> KnownModel {
    KnownModel {
        subscription_id: subscription.into(),
        model: name.into(),
        label: name.into(),
        description: description.into(),
        priority,
        efforts: vec!["low".into(), "medium".into(), "high".into(), "xhigh".into()],
        origin: "provider-list".into(),
        available: true,
        last_used: Some("2026-11-01".into()),
        usage,
        role: None,
    }
}

/// Learned knowledge covering roles from descriptions, family inheritance,
/// unavailable models, usage in neighbouring brackets and effort habits.
fn knowledge() -> Knowledge {
    Knowledge {
        state: "ready".into(),
        refreshed_at: Some("2026-11-12T00:00:00Z".into()),
        prompts_seen: 1270,
        sources: Vec::new(),
        subscriptions: vec![
            ObservedSubscription {
                id: "claude".into(),
                plan: Some("max".into()),
                listed_at: Some("2026-11-10".into()),
                last_used: Some("2026-11-11".into()),
                days_since_used: Some(1),
            },
            ObservedSubscription {
                id: "codex".into(),
                plan: None,
                listed_at: None,
                last_used: Some("2026-10-01".into()),
                days_since_used: Some(42),
            },
        ],
        models: vec![
            model(
                "claude",
                "golden-large-5",
                "Most capable model for complex, long-running work",
                Some(1),
                vec![
                    usage("complex", 40, 31.5, "high"),
                    usage("major", 12, 9.25, "xhigh"),
                ],
            ),
            model(
                "claude",
                "golden-mid-5",
                "Balanced everyday model",
                Some(2),
                vec![
                    usage("moderate", 90, 70.0, "medium"),
                    usage("simple", 20, 11.0, "low"),
                ],
            ),
            model(
                "claude",
                "golden-mid-4",
                "",
                Some(5),
                vec![usage("moderate", 3, 0.5, "medium")],
            ),
            model(
                "claude",
                "golden-small-5",
                "Fastest model for quick answers",
                Some(3),
                vec![usage("trivial", 50, 44.0, "low")],
            ),
            model(
                "codex",
                "golden-code-max",
                "Frontier agentic coding model",
                Some(1),
                vec![
                    usage("complex", 25, 18.0, "xhigh"),
                    usage("moderate", 15, 6.0, "high"),
                ],
            ),
            model(
                "codex",
                "golden-code-mini",
                "Smaller, faster and cheaper",
                Some(2),
                Vec::new(),
            ),
            KnownModel {
                available: false,
                ..model(
                    "codex",
                    "golden-code-old",
                    "Older model, superseded",
                    Some(9),
                    vec![usage("simple", 7, 1.0, "low")],
                )
            },
        ],
    }
}

fn bound_config() -> AssessmentConfig {
    let mut config = AssessmentConfig::default_alpha();
    config.structured_input = Some(StructuredInput {
        json_fields: vec!["title".into(), "description".into()],
        separator: "\n".into(),
    });
    config.model_score = Some(ModelScoreBinding {
        id: "golden-model".into(),
        manifest_sha256: "a".repeat(64),
        level_brackets: config.brackets.iter().map(|b| b.id.clone()).collect(),
        level_cut_points: vec![0.5, 1.5, 2.5, 3.5],
    });
    config
}

/// Deterministic stand-in model output: valid for most prompts, with invalid
/// and missing output mixed in so every model-score branch is pinned.
fn model_output(index: usize, text: &str) -> Option<ModelOutput> {
    if index % 11 == 5 {
        return None;
    }
    let seed = text.bytes().fold(index as u64 + 7, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as u64)
    });
    let raw: Vec<f64> = (0..5)
        .map(|level| ((seed >> (level * 7)) % 97 + 1) as f64)
        .collect();
    let total: f64 = raw.iter().sum();
    let mut probabilities: Vec<f64> = raw.iter().map(|value| value / total).collect();
    if index % 7 == 3 {
        probabilities[0] += 0.5;
    }
    Some(ModelOutput {
        probabilities,
        elapsed_micros: 1234,
        truncated: index % 5 == 0,
    })
}

fn digests(index: usize, text: &str) -> [String; 3] {
    let rules = rules();
    let plain = AssessmentConfig::default_alpha();
    let bound = bound_config();
    let learned = knowledge();
    let checks = BTreeMap::from([
        (
            "claude".to_string(),
            SubscriptionCheck {
                active: true,
                detail: "probe ok".into(),
            },
        ),
        (
            "codex".to_string(),
            SubscriptionCheck {
                active: false,
                detail: "signed out".into(),
            },
        ),
    ]);
    let request = |declared: &[&str]| AssessRequest {
        session_id: "golden".into(),
        revision: index as u64,
        text: text.into(),
        declared_subscriptions: declared.iter().map(|s| s.to_string()).collect(),
    };
    let digest = |mut assessment: lfg_core::assessment::Assessment| {
        assessment.duration_micros = 0;
        digest_value(&serde_json::to_value(&assessment).unwrap())
    };
    let output = model_output(index, text);
    [
        digest(assess_with_model(
            &plain,
            &rules,
            &request(&[]),
            &Knowledge::disabled(),
            &BTreeMap::new(),
            None,
        )),
        digest(assess_with_model(
            &plain,
            &rules,
            &request(&["claude", "codex"]),
            &learned,
            &checks,
            None,
        )),
        digest(assess_with_model(
            &bound,
            &rules,
            &request(&["claude"]),
            &learned,
            &checks,
            output.as_ref(),
        )),
    ]
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/assessment-golden.json")
}

#[test]
fn assessments_match_the_golden_digests() {
    let actual: Vec<[String; 3]> = PROMPTS
        .iter()
        .enumerate()
        .map(|(i, text)| digests(i, text))
        .collect();
    let path = fixture_path();
    if std::env::var_os("LFG_GOLDEN_WRITE").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&actual).unwrap() + "\n").unwrap();
        return;
    }
    let expected: Vec<[String; 3]> =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        expected.len(),
        PROMPTS.len(),
        "fixture and prompt list differ in length"
    );
    for (index, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(want, got, "assessment changed for golden prompt {index}");
    }
}

#[test]
#[ignore = "runs only with LFG_GOLDEN_PROMPTS and LFG_GOLDEN_OUT set"]
fn private_prompt_digests() {
    let input = std::env::var("LFG_GOLDEN_PROMPTS").expect("LFG_GOLDEN_PROMPTS");
    let output = std::env::var("LFG_GOLDEN_OUT").expect("LFG_GOLDEN_OUT");
    let mut lines = Vec::new();
    for (index, line) in std::fs::read_to_string(input).unwrap().lines().enumerate() {
        let row: serde_json::Value = serde_json::from_str(line).unwrap();
        let text = row["text"].as_str().unwrap_or_default();
        lines.push(digests(index, text).join(" "));
    }
    std::fs::write(output, lines.join("\n") + "\n").unwrap();
}
