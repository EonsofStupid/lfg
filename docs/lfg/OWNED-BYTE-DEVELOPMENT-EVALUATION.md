# Owned byte-model development measurement

`evaluate_owned_bytes.py` compares a distilled owned student, a cross-entropy
control with the same training configuration and source, a freshly initialized
student with that seed, and the pinned byte-GPT-2 teacher used by the candidate.
It verifies model and source identities, architecture and context compatibility,
and a different evaluation-file hash before running. A different hash is not
proof of independent authorship, semantic separation or absence of overlap.
The teacher's byte-ID mapping remains an operator assertion. Artifact manifests
are identity claims, not signatures or proof of data rights.

The input is an explicitly selected, already redacted local document. The tool
scores every subsequent byte once in consecutive, non-overlapping windows. Each
window starts with empty context, so results depend on the configured context
length and do not represent long-context generation. It reports next-byte
negative log-likelihood, bits per byte and exact byte accuracy. It also rotates
model order and records per-window CPU forward timings after warmup; these do
not include process startup, I/O, preprocessing or a persistent-service load
test. The short last window is included. The report contains hashes, identities
and aggregate results, not document text, but remains private and can reveal
information about the data. Outputs must be new and ignored and untracked inside
Git. Windows ACLs follow the parent directory. Training and evaluation inputs
must be authorized and secrets redacted before use.

The tool does not evaluate intent, complexity, model selection, effort or agent
count and does not run the deterministic LFG assessment baseline. Its report
sets `assessment_evaluated: false`, `independent_project_verified: false` and
`promotion_eligible: false`. Use a separately frozen, project-separated set of
actual assessment targets for the requested product evaluation.

## Local development observation, 2026-11-02

The training source was the LFG README. The separately selected evaluation file
was `docs/product-direction.md` from the owner's SWAG checkout at commit
`22558aa3b2f211854c8e5df86af02e0829e5432f`; the file was unchanged locally,
manually inspected and checked against the example local secret-pattern policy
with no matches. The two related projects and their histories have not been
audited for semantic overlap. This document has now been inspected and used for
development; it is not a blind final test. The reserved Makers-Impulse source
was not opened by this evaluation.

With a 64-byte context, 95 windows and 6,019 predicted bytes, two local CPU runs
produced identical loss and accuracy values:

| Model | Bits per byte, lower is better | Next-byte accuracy |
| --- | ---: | ---: |
| Initial random student | 8.376 | 0.17% |
| Cross-entropy control | 4.954 | 12.83% |
| Distilled student | 4.926 | 13.04% |
| Pretrained autocomplete teacher | 3.514 | 51.20% |

The small difference between the two students on one document is not evidence
of a reliable or practically meaningful improvement. This comparison says
nothing about assessment correctness, reasons, calibration or agent and model
recommendations. Teacher pretraining overlap with this document is unknown.
Per-window CPU p95 for the control varied from about 1.56 ms to 6.49 ms between
the two runs; the machine was not isolated. These numbers are diagnostics, not
qualified latency claims or representative user workloads.

The private reports, checkpoints, configurations and source hashes remain local.
The environment versions are recorded but not wheel-locked; cross-platform
reproduction has not been established.
