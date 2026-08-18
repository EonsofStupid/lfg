# LFG encoder experiment

This is the offline, non-authoritative training lane for the Rust
MasterClyffy CPU encoder. Runtime prompts are not collected here. The checked-in
corpus is intentionally authored synthetic data.

The promotion order is fixed:

1. Validate the corpus with `python3 train.py validate`.
2. Train with `python3 train.py train --base-model MODEL` only after the sample gate.
3. Export a versioned ONNX artifact and manifest.
4. Benchmark through the Rust runtime.
5. Promote only when `promotion-policy.json` passes.

Training requires `requirements.txt`. Validation uses only the Python standard
library and does not download a model.
