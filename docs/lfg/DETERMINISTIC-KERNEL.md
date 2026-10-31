# Experimental deterministic LFG kernel

The core library now has a separate `kernel::Engine` that compiles a JSON
snapshot into reusable regular expressions and evaluates named input fields.
It has no model, clock, randomness, network access, or storage in its evaluation
path. Its supported node types are Gate, Choice, Score, Extract and Compose.
The CLI is a thin caller of this library:

```powershell
cargo run --locked -p lfg-server -- classify "logout doesn't clear the session"
```

The bundled [snapshot](../../crates/lfg-core/config/kernel-core.v1.json) is a
small, explicitly authored example, not a trained classifier or a complete
project taxonomy. A different snapshot can be passed with `--snapshot PATH`;
its file size is bounded by the configured operational HTTP-body byte limit.
Additional inputs use repeated `--field NAME=VALUE` arguments. Input fields are
currently matched exactly as provided; the normalization and app adapters in
the research draft have not been implemented. The snapshot's input and regex
size limits, seed, nodes, rules, rule weights and thresholds are data.

For offline batches, `kernel-predict --max-examples N` reads JSONL records
containing `id` and `text` on stdin and emits one JSONL result per record. It
constructs the engine once for the batch, rejects duplicate IDs and stops at the
explicit limit. A nonzero exit means the batch did not complete, even if some
rows were already emitted. The bundled rules do not echo prompt text, but other
snapshots can extract text into results; protect input and output accordingly.
An optional [local journal](KERNEL-CORRECTION-JOURNAL.md) can record observations
and operator-attested choice corrections; it is not a reviewed training corpus.

Choice combines matching rule weights with integer fixed-point noisy-OR and
normalizes the resulting option scores. It escalates when nothing matches, the
configured threshold is not met, the margin is ambiguous, or any referenced
field is missing. These are **relative rule scores, not calibrated probabilities**:
one matching option can normalize to the maximum regardless of the absolute
rule weight. The scoring method and thresholds require real-data evaluation;
they are not a claim of model confidence. Gate prunes on a matching closed rule
and escalates on conflicting or absent evidence. Score uses an integer weighted
average; Extract takes the first matching rule's named capture. Compose finishes
the result. Extracted values appear in the output and can contain private input.

Results carry the snapshot digest and revision, an address, a seeded XXH3 key
computed from the address without its revision, and per-hop rule IDs, fields,
pattern digests and configured weights. Equal keys are not proof that distinct
paths cannot collide. Revisions must be managed immutably by the future
snapshot distribution system; this implementation does not enforce that policy.
Snapshot parsing rejects unknown fields, duplicate rules, missing targets,
cycles, invalid patterns and missing named captures. It compiles each rule once
per engine construction; the CLI currently constructs an engine per invocation.

The CLI test compares the complete output for the note's logout example to a
[golden file](../../crates/lfg-server/tests/fixtures/kernel-logout.v1.json).
The verification workflow is configured to save and compare that output as a
CI artifact, but GitHub's Actions budget currently prevents the job starting.
On 2026-10-31, the same golden file matched CLI output in debug builds on
Windows x86-64 (Rust 1.98.1) and the ARM64 Spark (Rust 1.97.1). The Spark ran
the five-operation core tests and the golden CLI test in a separate two-job
build directory. The shared snapshot digest was
`d715780edab8f4c9274328700264fa0f1d7ac8cebbf8a10bbc5a961566d28726`;
the golden file digest was
`9f6490f34d8f55c5cf24c68eb5a98a335676fcf07249a2a65dc029cb8cacb722`.
This establishes byte equality for this example, not general cross-platform
determinism. Local tests cover all five operations, explicit escalation, gate pruning,
ambiguous choices, invalid graphs, changing snapshots, repeatability and key
stability across revision changes. These are engineering tests, not evidence
of accuracy, throughput, representative coverage or cross-machine reproducibility.
The Spark also passed the later external-snapshot and bounded JSONL streaming
tests in a separate offline build; it did not receive the private requests.

The bounded stream was also run on 11 previously inspected LFG conversation
requests held in the ignored local review draft. None is human-verified. With
the bundled example snapshot, two returned `complete` and nine escalated for no
matching rule. This reports only the engine's states, not whether the two
addresses are correct, model quality, or representative coverage. No training
or review decision occurred. The private source file SHA-256 was
`18b51e3263550b389593c527278edec240d65129f404500ea026a078be332d3e`;
the Windows debug executable SHA-256 was
`bc7288df95e97a0af6c4e9fac07b842a4028e4e9153ed42508aa2f5d450db086`.
An optimized Windows build with its UI embedded also matched the golden, served
the UI, and returned the same aggregate states on those 11 requests. That build
was 7,607,808 bytes with SHA-256
`b252a483ecca5aa834fbae77010a8b9fa2e667fbb667cd6b2f995a4209dd6752`.
It is a local build smoke check, not a qualified or reproducible release.

The binary snapshot format and compiler, atomic swap, overlays, integrated
correction review, trained per-node or embedding extension, model promotion,
full app/HTTP integration, MCP/RRFlow/SWAG integration, and qualified release
remain to be built and evaluated. The existing legacy classifier endpoints are
not backed by this kernel. No project review decisions are created by it.
