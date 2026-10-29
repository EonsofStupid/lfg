# Written by Clyffy

The lfg project is a Rust workspace (crates: lfg-core, lfg-codex, lfg-runtime,
and lfg-server) that builds an axum/tokio-based server application.

The objective is a self-contained, CPU-oriented binary and UI that assess prompt
complexity and orchestration risk before execution, make the signals and task
boundaries visible, and suggest the model class, agent count, and reasoning depth
an operator or downstream system may need. LFG is an operational assessment and
routing layer for a larger reasoning workflow; it does not provide the final
answer in place of that workflow.

The current encoder is rule-based. Explicit protocol checks and evaluation
gates establish the baseline before learned or adaptive behavior can become
authoritative. The offline classifier and CPU LLM work are experiments toward
that objective, not evidence that a learned model is already in the runtime.
See the project objective and current status in [README.md](README.md).
