# LFG Release Checklist

Complete this checklist before creating a signed `v*` tag.

## Product and behavior

- [ ] Assessment overlay behavior is still correct (no accidental answer-substitution behavior).
- [ ] UI strands/mapping and complexity decomposition views are functionally validated.
- [ ] Model recommendation and task-separation heuristics were spot-checked against recent scenarios.

## Build and validation

- [ ] `scripts/pre-release-check.ps1` passed without failures.
- [ ] Rust workspace checks and tests passed.
- [ ] ML validation scripts passed (`train.py validate`, `distill.py` CLI).
- [ ] UI production build passed.
- [ ] Deployment image build passed and health endpoint is green.

## Supply chain and provenance

- [ ] OCI metadata labels are present (`version`, `revision`, `created`).
- [ ] SBOM was generated and reviewed for unexpected packages.
- [ ] High/Critical vulnerability gate status is acceptable.

## Release packaging

- [ ] Version/tag follows semantic versioning (`vMAJOR.MINOR.PATCH`).
- [ ] Release notes are accurate for operators and downstream integrators.
- [ ] Artifact signatures (cosign) and checksums are published with release artifacts.
