//! Agents: how many to run, from the separate tasks found, kept within the
//! chosen bracket's range.

use serde::{Deserialize, Serialize};

use super::config::Bracket;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentResult {
    pub total: usize,
    pub tasks: usize,
    pub min: usize,
    pub max: usize,
    pub plain: String,
}

/// One agent per separate task, at least one, clamped to the bracket's range.
pub(super) fn agents_for(bracket: &Bracket, tasks: usize) -> AgentResult {
    let total = tasks.max(1).clamp(bracket.min_agents, bracket.max_agents);
    let range = if bracket.min_agents == bracket.max_agents {
        format!("{}", bracket.min_agents)
    } else {
        format!("{}–{}", bracket.min_agents, bracket.max_agents)
    };
    let plain = format!(
        "{tasks} separate task{} found. {} work uses {range} agent{}, so LFG suggests {total}.",
        if tasks == 1 { "" } else { "s" },
        bracket.label,
        if bracket.max_agents == 1 { "" } else { "s" },
    );
    AgentResult {
        total,
        tasks,
        min: bracket.min_agents,
        max: bracket.max_agents,
        plain,
    }
}
