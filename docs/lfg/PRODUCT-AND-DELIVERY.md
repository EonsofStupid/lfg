# LFG product and delivery target

LFG should be a self-contained, CPU-oriented binary that users can run on their
own machine or host as a service. The binary should provide local analysis and
an optional developer UI, connect to supported AI tools, and integrate with
cloud services when the user configures them. Running locally must not require
an account, cloud connectivity, RRFlow, SWAG, a GPU, or a model download.

This is the target, not a statement that a fully qualified self-contained
distribution, hosted authentication, cloud integration, or model editor exists
today. An optional build feature embeds the separately built UI into the server;
see [bundled UI builds](BUNDLED-UI.md) for evidence and limits. The classifier
is rule-based. Training is experimental.

## Developer UI

The UI should show bounded traces, errors, latency and real resource measures,
effective configuration and artifact identities, and make performance changes
observable. It must label elapsed time, CPU time, cycles and unavailable values
correctly. Diagnostics should exclude secrets and prompt contents by default,
with configurable collection, retention and export. Hosted deployments require
authenticated, authorized access to administrative and diagnostic endpoints.

An operator should be able to create, inspect, edit, validate and test a
project's LLM integration through the developer UI: connection type, provider
and model identifiers, supported effort options and capabilities, endpoint,
limits, routing policy, and local or cloud availability. These are versioned
configuration, not a provider roster or model policy embedded in executable
code. The available fields and allowed values must reflect the selected adapter
and its validated capability schema. Provider-specific features or account
entitlements that cannot be verified must be shown as unknown.

Credentials belong in an appropriate secret store and configuration holds
references rather than credential values. A change needs validation, a preview
of its effective scope and consequences, and an explicit apply; failures leave
the last valid configuration active. Users need export, import, audit and
rollback appropriate to deployment mode. Project settings must not be
overwritten by global catalog or model updates. Connection tests must disclose
when they can send data or incur usage. A selection, AI recommendation or
subscription declaration cannot grant permissions or silently dispatch work.

The existing Inspector shows recent metadata-only traces and adjusts browser
debounce/history for that session. It does not edit server-side model
configuration, store credentials, manage connections, or measure CPU cycles.

## Integration and release boundaries

The standalone binary should expose the same versioned analysis through its
local UI and a typed MCP adapter. RRFlow/SWAG integration is optional: LFG
contributes observations; RRFlow controls authorization and budgets; SWAG
proposes model and effort. Cloud transport, identity, sharing and synchronization
must be explicit, configurable and independently authorized. Local operation
must remain useful when those integrations are unavailable.

Distribute a build for each qualified OS and architecture with embedded or
otherwise verified bundled UI assets, tested installation and upgrade, and no
dependency on a source checkout. Version and verify binary, rules, provider
catalogs and trained artifacts independently, with compatibility checks and
rollback. Measure binary size, startup, memory, elapsed latency, CPU time, and
where available CPU cycles on declared hardware and a representative corpus.
Publish quality and resource limits alongside benchmark methodology.

The first implementation milestones are: (1) reproducible, reviewed data and a
CPU baseline; (2) reproducible checkpointed training and independent evaluation;
(3) qualified artifact export and CPU integration; (4) embedded UI and clean
machine release tests; (5) validated project configuration APIs and developer
editor with secure credential references; and (6) MCP, RRFlow/SWAG and cloud
adapters tested through their own access and failure boundaries. These tracks
can advance independently where their contracts are stable; none is complete
merely because a local smoke test or GPU run succeeds.
