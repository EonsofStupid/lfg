# Written by Clyffy

The lfg project is a Rust workspace (crates: lfg-core, lfg-codex, lfg-runtime,
and lfg-server) that builds an axum/tokio-based server application.

The objective is LFG as a tuned model and a standalone product: a
self-contained, CPU-oriented binary and UI that, for each prompt, report the
intent, complexity, suggested models and total agents, with the evidence behind
each. It is useful on its own and can be combined with SWAG and RRFlow. SWAG
accepts LFG's assessment as a suggestion and weights it more heavily after a
configurable confirmation, once LFG's suggestions are proven reliable. LFG does
not grant authorization or produce the final answer.

LFG's model is our own: distilled and fine-tuned on the owner's projects and
chats, about 650 MB, and bound into one binary that runs locally or as a
network service. It is built by applying Liquid AI's published efficiency
research rather than adopting their models. Users start from LFG's baseline and
can train custom weights on their own successful chats and projects. The
current encoder is rule-based and the training lane does not yet do this; see
[model and distillation](docs/lfg/MODEL-AND-DISTILLATION.md) and the current
status in [README.md](README.md).
