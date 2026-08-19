# LFG prompt-recipe encoder

This is the offline, non-authoritative training lane for the Rust
MasterClyffy CPU preflight encoder. It learns multi-label intent, capability,
and risk classification. It is not the Jina/GTE retrieval lesson and it does
not write the final CoSTAR packet. Runtime prompts are not collected here. The
checked-in corpus is intentionally authored synthetic data.

Training is GPU-only. CPU is reserved for the exported ONNX/Rust proof; the
trainer refuses to fall back to CPU. Unsloth is the default engine, using its
current `FastModel` sequence-classification path with full fine-tuning. The
plain Transformers engine is available only as an explicit compatibility
diagnostic.

The promotion order is fixed:

1. Validate the corpus with `python3 train.py validate`.
2. Review the immutable plan with `python3 train.py plan`.
3. Train with `python3 train.py train` only after the 500-human-verified sample gate.
   A tiny `--development-run` is allowed for compatibility work, but its artifact
   is permanently marked ineligible.
4. Export a versioned ONNX artifact and manifest.
5. Benchmark through the Rust runtime.
6. Promote only when `promotion-policy.json` passes.

The base is pinned to `answerdotai/ModernBERT-base` revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. Training requires
`requirements.txt`. Validation and planning use only the Python standard
library and do not download a model.

```powershell
python train.py validate
python train.py plan
python train.py train --output artifacts\run-001
python train.py export --output artifacts\run-001
python train.py benchmark --output artifacts\run-001 --max-length 512

# Compatibility proof only; never promotable:
python train.py train --development-run --epochs 1 --output artifacts\compat-001
```
