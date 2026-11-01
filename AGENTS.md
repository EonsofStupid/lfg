# Direction for AI agents working on LFG

Read this before planning work. It is the owner's direction.

- **LFG is our own model and a standalone product.** For each prompt it reports
  intent, complexity, suggested models and total agents, with the reasons. It
  ships as one clean binary with the runtime, a model of about 650 MB, and a
  layman-friendly UI. It runs locally or as a network service, and can be
  combined with SWAG and RRFlow.
- **Distill from the owner's projects.** The owner authorizes using their own
  projects and chats for training and distillation. Keep that data local and
  out of Git, and redact secrets.
- **Learn from research, build our own.** Apply Liquid AI's published efficiency
  research: hardware-in-the-loop architecture search, short-convolution hybrids,
  Top-K distillation, and RIPL's real-time latency targets (the owner's
  hypothesis is that RIPL uses a distinctive KV-cache design and organized
  shards). Do not adopt or rebrand their models, or anyone else's, as LFG.
- **Organizational knowledge makes it hit above its weight.** Build the record
  of relationships between projects, prompts, models, agents and outcomes.
  Hold it in organized KV-cache shards, and periodically fine-tune stable
  knowledge into the weights.
- **Gates decide trust, not whether to build.** Evaluation gates decide when a
  model version or suggestion is trusted more. They must not block training on
  authorized data or block the product.
- **Users adapt it.** Every user gets the baseline and can train custom weights
  on chats and projects they mark successful. Confirmation and weighting rules
  are configurable.

Details: [README objective](README.md#project-objective),
[model and distillation](docs/lfg/MODEL-AND-DISTILLATION.md),
[product and delivery](docs/lfg/PRODUCT-AND-DELIVERY.md).
