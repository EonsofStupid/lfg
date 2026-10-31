# Baseline and candidate evaluation smoke

The evaluation workflow was exercised on the 25-row RRFlow review draft. Every
row is unverified, reconstructed from project material, and subject to the
labeling uncertainties in [the labeling guide](../../ml/lfg-encoder/LABELING.md).
This is an exploratory wiring and error-analysis run, not a representative or
independently reviewed final evaluation. The set has no positive
`sensitive-data` examples, so security recall is unavailable.

The baseline is the Rust `lfg-cpu-rule-baseline` configuration with digest
`7302d5e88738ab686f98e17c701303ee5020aa343007aa1eb3bf6bdf5af4e032`.
The candidate is the local CPU reload of the tiny random BERT plumbing model
trained on the six synthetic examples; the development threshold was 0.5.
Its identity digest for this run was
`5bd56e6babab9a09a6a7822e0151154d5c9e247d66d59029dd5e394c75d720a5`.
The candidate is ineligible for production and is not evidence of what a
properly trained model can achieve.

Both were scored against dataset SHA-256
`ca53cbba34419f666373efb87872c7eff4dd88367ae27f53de9c9dba1a665dbc` and
taxonomy SHA-256
`7e935b156e9eb56c5f00b7e68cbb2d0d77234f73cc8ac844abb460f19a2a9e9f`.
The declared IDs and family names did not overlap the candidate's recorded
development split; that check does not establish true independence.

| Exploratory metric | Rust baseline | Tiny test model |
| --- | ---: | ---: |
| Intent macro F1, four supported labels | 0.663 | 0.000 |
| Capability macro F1, four supported labels | 0.663 | 0.069 |
| Sensitive-data positive examples | 0 | 0 |

Macro F1 includes only labels with positive reference support; micro F1 and
per-label errors are in the local reports. Prediction output hashes were
`6bc8614f92ca4c24886c69395bd1c621871c0ce79e5d2c7568748c2ff234f3fd`
(baseline) and
`bc223930dcd60c9f95faed5de01cc55cd2a372d8b832b60d58d4141136d3e3db`
(candidate). Both reports are marked ineligible for promotion. Results,
prediction records and the experimental model remain in ignored local build
output; they are not deployed. The comparison does not measure calibration,
CPU time, hardware cycles, memory, or a release binary.

A separate local-only diagnostic used the five unverified, related Makers
Impulse prompt examples described in the labeling guide. On those draft labels,
the baseline's intent and capability macro F1 were 0.2 and 0.4; the tiny test
model scored 0 for both. These five examples share a family, were already
inspected, and cannot serve as a blinded final test. No private prompt text or
individual prediction records are included in this document.

## Real-source DevPulse diagnostic

On 2026-10-30, a Windows x86-64 release binary built from commit `36166d2`
ran the deterministic baseline on 12 attributed DevPulse excerpts. Their
source spans were verified against pinned project documents, but all reference
labels were proposed by an AI and remain unreviewed. All 12 were provisionally
labeled `build.software` and `filesystem.write`; the baseline returned each of
those labels for two of them. This is a disagreement with a provisional draft,
not a measured error rate or model-quality claim. No positive security-risk
examples were present. Predictions and the report remain in the ignored
private dataset directory.

The baseline configuration digest was
`7302d5e88738ab686f98e17c701303ee5020aa343007aa1eb3bf6bdf5af4e032`; the
dataset digest was `e62fc45f55eabaac8b253fa0935c4bcb36e1debf964a0b37f8f6eb2c08a6425a`.
The executable digest was
`42b096f319f1a5143c6eb0760121e3f516bf1de9371f8e13e9a642e809067a84`.
It was built offline with the lockfile, embedded UI, `+crt-static`, and `/Brepro`
flags after building the UI with the existing local Node dependencies. A copy
of the 4,994,560-byte executable started from a separate local directory and
served its embedded UI and intake template. This is a local smoke check, not a
clean-machine installation, reproducible-build comparison, performance
benchmark, trained-model integration, or hosted security qualification.
