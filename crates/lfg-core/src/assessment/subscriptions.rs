//! Which subscriptions the user has, and which may be used for suggestions.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::AssessRequest;
use super::config::AssessmentConfig;
use super::knowledge::{Knowledge, SubscriptionCheck};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionStatus {
    pub id: String,
    pub label: String,
    /// `verified`, `recognized`, `declared`, `signed-out` or `not-found`.
    pub status: String,
    pub usable: bool,
    pub declared: bool,
    pub plan: Option<String>,
    pub last_used: Option<String>,
    pub detail: String,
}

pub(super) fn subscription_statuses(
    config: &AssessmentConfig,
    request: &AssessRequest,
    knowledge: &Knowledge,
    checks: &BTreeMap<String, SubscriptionCheck>,
) -> Vec<SubscriptionStatus> {
    let mut ids: BTreeSet<&str> = config.subscriptions.iter().map(|s| s.id.as_str()).collect();
    ids.extend(knowledge.subscriptions.iter().map(|s| s.id.as_str()));
    ids.extend(request.declared_subscriptions.iter().map(String::as_str));
    ids.into_iter()
        .map(|id| {
            let label = config
                .subscriptions
                .iter()
                .find(|s| s.id == id)
                .map_or(id, |s| s.label.as_str());
            let observed = knowledge.subscriptions.iter().find(|s| s.id == id);
            let declared = request.declared_subscriptions.iter().any(|d| d == id);
            let check = checks.get(id);
            let plan = observed.and_then(|s| s.plan.clone());
            let last_used = observed.and_then(|s| s.last_used.clone());
            let recently_used = observed
                .and_then(|s| s.days_since_used)
                .is_some_and(|days| days <= config.matching.seen_within_days);
            let listed = observed.is_some_and(|s| s.listed_at.is_some());
            let recognized = plan.is_some() || recently_used || listed;
            let used_text = last_used
                .as_deref()
                .map(|day| format!("last used {day}"))
                .unwrap_or_else(|| "no recent sessions".into());
            let (status, detail) = match check {
                Some(check) if check.active => (
                    "signed-in",
                    format!("Signed in; subscription entitlement is not verified; {used_text}."),
                ),
                Some(check) if !declared => ("signed-out", check.detail.clone()),
                _ if recognized => ("recognized", format!("Found on this machine; {used_text}.")),
                _ if declared => (
                    "declared",
                    "You said you have this; LFG could not confirm it.".to_string(),
                ),
                _ => ("not-found", "Not found on this machine.".to_string()),
            };
            SubscriptionStatus {
                id: id.to_string(),
                label: label.to_string(),
                status: status.into(),
                usable: declared,
                declared,
                plan,
                last_used,
                detail,
            }
        })
        .collect()
}
