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
- **The first dev model can build on open models.** For the owner's own
  deployment, the first dev model may be distilled from or built on open models
  whose licenses allow it, so LFG can start developing itself alongside a dev
  RRFlow. Use the owner's Sparks for the next few weeks and rent hardware only
  for gaps measured there. A fully owned model remains the goal, and nothing
  released to others may present another model as LFG's own. See the
  [first dev model](docs/lfg/FIRST-DEV-MODEL.md).
- **Organizational knowledge makes it hit above its weight.** Build the record
  of relationships between projects, prompts, models, agents and outcomes.
  Hold it in organized KV-cache shards, and periodically fine-tune stable
  knowledge into the weights.
- **Model knowledge is learned, not hand-kept.** Which models exist, what they
  are for and how they are used comes from provider tools' own data and the
  user's sessions, and updates as they work. Do not add hand-maintained model
  catalogs or leave models unrated. See [one assessment](docs/lfg/ASSESSMENT.md).
- **Gates decide trust, not whether to build.** Evaluation gates decide when a
  model version or suggestion is trusted more. They must not block training on
  authorized data or block the product.
- **Users adapt it.** Every user gets the baseline and can train custom weights
  on chats and projects they mark successful. Confirmation and weighting rules
  are configurable.

## Execution rules

- Advance the actual product milestone: representative authorized data,
  meaningful training, task evaluation, and a usable integrated build. A tiny
  pilot or fixture establishes only the specific path it exercises; do not make
  another isolated pilot the default next milestone after that path works.
- Training may use selected authorized data before review or quality thresholds
  are met. Record provenance and uncertainty, and apply evaluation gates to
  trust and promotion. Never describe a teacher proposal as a verified outcome.
- Set architecture, data, resource and integration policies through configuration.
  Accept extension fields where the contract permits them; do not imply an
  unrecognized field changes behavior. Size Spark work from measurements and
  coexistence needs rather than retaining a pilot's resource cap by habit.
- Keep private inputs, logs and derived weights local and out of Git. Redact
  before use, retain source identities, and keep independent evaluation material
  separate. Screening and labels are claims unless independently verified.
- Run the checks needed for the change and report their actual scope. While
  checked merges or builds run, continue independent substantive work; reconcile
  moving bases and finish authorized merges without bypassing required checks.
- Report the real blocker and the next action when the end state is incomplete.
  Do not present scaffolding, synthetic fixtures or a passing smoke test as a
  trained, qualified product.

Details: [README objective](README.md#project-objective),
[model and distillation and the order of work](docs/lfg/MODEL-AND-DISTILLATION.md#order-of-work),
[product and delivery](docs/lfg/PRODUCT-AND-DELIVERY.md).
