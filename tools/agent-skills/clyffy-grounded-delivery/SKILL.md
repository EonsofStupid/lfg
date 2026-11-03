---
name: clyffy-grounded-delivery
description: Ground substantial Clyffy project work in current evidence, create a verifiable implementation handoff, and route bounded execution with escalation. Use for LFG, RRFlow, SWAG, or related project work when the owner requests a grounded plan, model-tier handoff, or sustained implementation; do not invoke for unrelated simple tasks.
---

# Clyffy grounded delivery

Read the current user's instructions, repository `AGENTS.md`, actual working-tree state and relevant product owners before choosing work. Preserve user changes and the real product objective. Distinguish implemented, observed, planned and unknown behavior; cite exact source or measurement. Research current external technical claims from primary sources when needed. Do not invent data, outcomes, model capability or quality.

For a substantial task, use the configured **planner** route for grounding and the **executor** route only for bounded, executable work with clear ownership, dependencies and observable acceptance. Read [routing and hooks](references/routing-and-hooks.md); model names and efforts are configuration, not model performance guarantees. When the host supports explicit subagents and delegation is authorized, request the configured model and effort at spawn. Otherwise use the documented CLI adapter or continue with the available model and disclose the limitation; never claim the parent model was switched by a skill or hook.

Build a handoff using [the contract](references/handoff-contract.md) and validate it with `scripts/handoff.py validate`. Record the exact repository revision, evidence, permitted scope, no-go boundaries, work items, dependencies, checks, acceptance and unresolved facts. Use paths and digests instead of private prompt content, credentials, datasets, or weights. A structurally valid handoff is not proof that facts are true or work is complete. A newer user instruction overrides a stale plan.

Dispatch only the next ready work item; validate anchors immediately beforehand. Give the worker the actual handoff and require a structured result: files changed, exact commands and results, observed behavior, unresolved risks and next action. Integrate dependent steps in order and run the appropriate checks. Review the result against the handoff and repository state. Escalate to the planner for changed scope, stale or conflicting evidence, unclear requirements, repeated failures, unexpected side effects, security-sensitive choices, or a requested capability the host does not support. Do not quietly keep retrying or substitute a fixture for product evidence.

Track actual model/effort, handoff revision, route, elapsed time, check outcome, rework and blockers where the host exposes them; record unavailable usage/cost as unknown. Compare real tasks over time rather than assuming the cheaper route is adequate. Training on authorized data is not blocked by quality gates; gates govern trust and promotion. The user can pause, cancel or revise the work at any time.
