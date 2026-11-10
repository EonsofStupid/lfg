# First dev model

Owner direction, 2026-11-03: for the owner's own deployment, take a model suited
to NVIDIA hardware and the Sparks, distill a first official version quickly, spend
the next few weeks seeing what can be accomplished locally, and rent hardware only
for gaps measured there. The aim is a working dev LFG that improves through use
alongside a dev RRFlow. A fully owned model remains the goal, and nothing released
to others may present another model as LFG's own.

This does not replace the [grounded plan](GROUNDED-EXECUTION-PLAN-2026-11-03.md). Its
feedback journal is what lets dogfooding produce corrections. Treat this dev model as
the first candidate in that plan's model and hardware steps.

## Candidates

| Model | Size | Why it is a candidate | Limits |
|---|---|---|---|
| [Bonsai 4B](https://huggingface.co/prism-ml/Bonsai-4B-gguf) | 572 MB, 1-bit | Fits the 650 MB target with 78 MB to spare; Apache 2.0 | Needs PrismML's runtime fork for its kernels. Quality on LFG's inputs is unmeasured |
| [Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF) | 2.84 GB, 4-bit | Lists DGX Spark as a target; four attention layers should keep its KV cache small | Over the size target. Its [license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/) allows derivatives and does not directly address distillation |

The intended first version is a frozen 1-bit base plus a separately stored adapter of
up to 78 MB. [Edge0](https://arxiv.org/abs/2609.18063) kept an adapter separate from a
frozen 4-bit base, and merging it erased most of the gain. No published test on a 1-bit
base was found, so this is the main technical unknown. If it fails, a smaller base such
as the 429 MB [Ternary Bonsai 2 2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf)
leaves more room for the adapter.

## Evaluation

On 2026-11-03 no reviewed held-out set covering intent, complexity, models and agent
count was found. The first comparison can only measure agreement with the 470 unreviewed proposals from
the larger local teacher, plus speed and memory on the actual Spark. That shows how
far a small model is from its teacher, not whether either is correct. Validation and
test files stay closed to this collection. Accuracy claims wait for owner corrections
and confirmed outcomes.

## Per-project weights

The owner's theory is that once a project passes a threshold, RRFlow traces can build
weights for that project, benefiting local and frontier use without spending
subscription usage on what is answered locally. Frontier models cannot load these
weights, so they benefit only through fewer and smaller calls.

- [Code2LoRA](https://arxiv.org/abs/2606.06492) generated repository-specific adapters
  from snapshots and diffs, tested on Python assertion completion only.
- [TokenEconomizer](https://arxiv.org/abs/2610.22732) reduced frontier coding-agent
  token cost by 18.4–30.9% from per-repository traces without changing weights.

No published threshold was found. Train regardless, and use an adapter once it beats
the base on that project's later work.
