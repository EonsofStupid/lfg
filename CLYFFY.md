# Written by Clyffy

The lfg project is a Rust workspace (crates: lfg-core, lfg-codex, lfg-runtime,
and lfg-server) that builds an axum/tokio-based server application.

The objective is LFG as a tuned model and a standalone product: a
self-contained, CPU-oriented binary and UI that, for each prompt, report the
intent, complexity, suggested models and total agents, with the evidence behind
each. It is useful on its own and can be combined with SWAG and RRFlow. SWAG
accepts LFG's assessment as a suggestion and weights it more heavily after a
second confirmation, once LFG's suggestions are proven reliable. LFG does not
grant authorization or produce the final answer.

The current encoder is rule-based, and the training lane labels only intent,
capability and risk. Tuning LFG's model toward all four results and shipping it
as the binary product are the main lines of work; evaluation gates decide when
a tuned version is trusted. See the project objective and current status in
[README.md](README.md).
