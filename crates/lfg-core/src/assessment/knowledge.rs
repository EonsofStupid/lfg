//! What the host has learned about subscriptions and models, from provider
//! tools and the user's sessions. Counts and identifiers only, never prompt text.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// What the host has learned about available models and how they are used.
/// Contains counts and identifiers only, never prompt text.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Knowledge {
    pub state: String,
    pub refreshed_at: Option<String>,
    pub prompts_seen: u64,
    pub sources: Vec<KnowledgeSource>,
    pub subscriptions: Vec<ObservedSubscription>,
    pub models: Vec<KnownModel>,
}

impl Knowledge {
    pub fn disabled() -> Self {
        Self {
            state: "disabled".into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeSource {
    pub id: String,
    pub files: usize,
    pub prompts: u64,
    pub status: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservedSubscription {
    pub id: String,
    pub plan: Option<String>,
    /// When the provider's own model list was last fetched by its local tool.
    pub listed_at: Option<String>,
    pub last_used: Option<String>,
    /// Whole days since `last_used` when this knowledge was refreshed.
    pub days_since_used: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnownModel {
    pub subscription_id: String,
    pub model: String,
    pub label: String,
    pub description: String,
    pub priority: Option<u32>,
    pub efforts: Vec<String>,
    /// `provider-list`, `session` or `published`.
    pub origin: String,
    /// False when a provider list exists and no longer offers this model.
    pub available: bool,
    pub last_used: Option<String>,
    pub usage: Vec<BracketUsage>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BracketUsage {
    pub bracket: String,
    pub prompts: u64,
    /// Prompt count discounted by age, so recent work counts for more.
    pub weighted: f64,
    pub efforts: BTreeMap<String, u64>,
}

/// The result of an explicit sign-in check, such as a provider CLI probe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubscriptionCheck {
    pub active: bool,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeSummary {
    pub state: String,
    pub refreshed_at: Option<String>,
    pub prompts_seen: u64,
    pub models_known: usize,
}
