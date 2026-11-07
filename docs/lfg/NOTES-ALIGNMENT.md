# Alignment to the recovered LFG notes

The owner-authored `LFG_NOTES_v3` (2026-09-26, a working draft containing both
locked and proposed decisions) supersedes the earlier v1, v2 and v2.1 notes.
`LFG_ENGINE_PLAN` is a researched draft, not a completed implementation. The
later boundary question bank and stack drafts are likewise drafts. Treat these
as project source material, not proof that the described code already exists.

The owner's current direction is that LFG is also its own fine-tuned
classification and assessment model for development work, including intent,
complexity, model and agent recommendations, and CPU-burst efficiency. The
target is an approximately 650 MB model in a standalone product. The recovered
deterministic design remains a useful baseline and source of provenance; it
does not replace that model objective. CPU efficiency requires measurements on
representative hardware and is not established by a training label.

The v3 notes describe LFG as a deterministic System 1 classification kernel
and a separate extension. The kernel has Gate, Choice, Score, Extract and
Compose operations; identical input and snapshot revision must produce
byte-identical output with rule provenance. The extension may use embeddings,
per-node classifiers or external model answers when the kernel escalates; its
suggestions require evaluation, and proven results can become candidate rules.
Governance is separate, RRFlow owns durable data, and the runtime must not
silently grant authorization. The notes describe a bundled, offline first
binary and an eventual server mode using the same engine.

The repository now includes an [experimental deterministic kernel](DETERMINISTIC-KERNEL.md)
with all five node types and a CLI golden for the note's example. Its small JSON
snapshot, rule scoring and tests are an initial implementation, not the full
compiled snapshot and promotion architecture. The global multi-label classifier,
corpus review tooling, and ONNX experiments remain separate experiments; they
do not implement the exemplar-based extension or rule promotion chain and do not
establish model quality. The prior real-data work and hardware probes remain
useful evidence about their own limited scope.

The engine plan's first proof target is a real CLI classification from an
embedded core tree to one exact address and rule evidence, compared with a
golden artifact. That local CLI comparison now exists; the CI artifact cannot
yet be produced while Actions is blocked. Before using learned suggestions in
that chain, define the
per-node outcomes and reviewed journal data, establish a genuinely independent
evaluation split, and compare against the deterministic kernel. The notes
explicitly distinguish a correction journal from fabricated labels. The current
13-label taxonomy and review queues do not by themselves provide those per-node
outcomes. No review or quality gate is waived by this alignment.

The recovered [boundary question bank](BOUNDARY-QUESTION-BANK.md) supplies the
owner's actual question drafts and project-artifact names. This is governance
content to be integrated at an appropriate boundary, not a new LFG kernel or an
implemented oracle. Proposed maturity and stack routing still need to be built
and verified. The original notes also contain open design questions; their draft
wording should not be promoted to an irreversible policy without evidence.
