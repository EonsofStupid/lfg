# Two-checkpoint source-byte evaluation

`evaluate_owned_checkpoint_pair.py` compares two completed `lfg.owned-distillation-result.v1` checkpoints that use the same owned byte-student implementation and architecture. It scores both on one explicitly selected local UTF-8 file after applying the supplied private corpus screening policy. The report records original and screened SHA-256 identities and aggregate next-byte NLL, bits per byte, perplexity, accuracy, predicted bytes, windows, CPU timing, and right-minus-left metric deltas. It does not include source text or model-generated text.

The evaluator verifies each manifest and safetensors digest independently, the supplied per-checkpoint student-config file and digest, current `owned_student.py` digest, finite weights, architecture, parameter limit, and each checkpoint's own source-member and declared-validation identities. It rejects evaluation source digests matching either checkpoint's declared source or validation members. These metadata are claims from each checkpoint, not independent proof of project separation. The report sets project and semantic independence to `unknown`, marks assessment quality as unevaluated, does not compare training histories, and is never promotion-eligible.

Use a private source copy that has been screened and independently selected for evaluation. Configure limits in `owned-checkpoint-pair-evaluation.example.json`; pass a screening policy compatible with `private-corpus-screen.example.json`. The output must be a new file outside tracked Git content or ignored and untracked. On POSIX, the evaluator checks the output parent's owner-only permissions, rejects unsafe writable ancestors and symlink components, and creates the report with mode 0600. It fails closed on Windows because this utility has no verified owner-only ACL writer. No training, checkpoint writes, or promotion occur.

Example invocation:

```text
python evaluate_owned_checkpoint_pair.py \
  --config /private/evaluation-config.json \
  --source /private/selected-evaluation.txt \
  --screening-policy /private/screening-policy.json \
  --output /private/new-report.json \
  --left-manifest /private/checkpoint-a/manifest.json \
  --left-weights /private/checkpoint-a/weights.safetensors \
  --left-student-config /private/checkpoint-a/student-config.json \
  --right-manifest /private/checkpoint-b/manifest.json \
  --right-weights /private/checkpoint-b/weights.safetensors \
  --right-student-config /private/checkpoint-b/student-config.json
```

Byte loss and perplexity measure next-byte modeling only. They are not routing accuracy, recommendation quality, or evidence that the selected source is semantically or project-level independent of training data.
