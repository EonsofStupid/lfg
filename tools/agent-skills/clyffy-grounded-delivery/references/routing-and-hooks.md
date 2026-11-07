# Routing and hooks

Use `routing.toml` in the installed skill, or pass `--policy` to the handoff
tool. The provided example reflects the owner's requested Terra xhigh planning
and Luna execution, with Terra escalation. Availability, access, effort names,
price and latency depend on the host and can change. No fallback model is
silently substituted. Tune the executor effort from measured outcomes.

Native Codex custom-agent files can set `model` and `model_reasoning_effort`;
the host must expose subagent delegation. A skill cannot switch its own active
turn. The optional CLI dispatch invokes a new `codex exec` process with the
selected model and effort; it inherits the caller's existing configuration and
does not grant extra permissions. It validates the handoff and file anchors
before starting. Its dry run is the default; `--execute` starts real work and
must remain within the user's authorization. No automatic retry or model
fallback is performed.

The companion hook emits only a brief skill reminder for an explicit skill
invocation or named Clyffy subagent. It does not route or launch models, read
private datasets, persist prompts, or force completion. Hooks require host
support and, for non-managed definitions, the host's trust review. Tool hooks
may be skipped, disabled or fail open; never treat them as a security or
completion boundary. The schema validator and any applicable repository and
host controls are separate. Do not create a stop-hook continuation loop.

Official references: [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents),
[Codex hooks](https://learn.chatgpt.com/docs/hooks), and
[agent workflow evaluation](https://developers.openai.com/api/docs/guides/agent-evals).

## Installation and invocation

From the tracked source, run `python scripts/install.py`. It installs a personal
skill, two named custom-agent files and an advisory `hooks.json` under
`$CODEX_HOME` (or `~/.codex`); it refuses to overwrite any of those paths. If
there is an existing hooks file, merge the two hook entries manually rather than
replacing it. Inspect and trust the hook through the host's `/hooks` interface;
restart or start a new session if discovery does not pick up the skill or agents.
Hook trust is never bypassed. Invoke `$clyffy-grounded-delivery` for this workflow.
Automatic skill discovery and subagent selection are host-dependent, not enforced.

The installed `routing.toml` governs CLI dispatch. The two agent TOML files are
separate host configuration and are initialized from that policy at installation;
edit both places when changing a route. Explicit native subagent spawning can also
use the policy's model and effort. Updates to the tracked source do not automatically
overwrite the installed personal copy. Review changes and update it deliberately.

To inspect without launching work, run `python scripts/handoff.py dispatch HANDOFF
--repo REPOSITORY`; add `--execute` only to launch the explicitly routed task.
The private event log records process lifecycle, not verified task success; review
the worker's evidence and run the checks. Native subagent calls do not automatically
write this dispatch log. Unknown token usage and cost stay null. The log must be
outside the repository, and the caller remains responsible for platform access controls.
For an executed CLI attempt, independently review the result and record the outcome
against the original unchanged handoff using `python scripts/handoff.py record HANDOFF
--repo REPOSITORY --attempt ATTEMPT_UUID --result accepted --checks-passed N
--checks-failed N --rework-count N` (add `--escalated` if applicable). Use `rejected`
or `blocked` as appropriate. Counts and results are operator attestations, not
automatically verified evidence. The attempt UUID is printed and written to the
private log; started and exited events do not themselves mean task success.
