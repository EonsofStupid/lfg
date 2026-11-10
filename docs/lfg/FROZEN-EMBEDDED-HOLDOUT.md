# Frozen embedded candidate and held-out comparison

`evaluate_embedded_holdout.py` separates three local actions: a holdout
custodian commits the identities of reviewed test files, an operator freezes a
development candidate against that commitment, and a final evaluator opens the
committed test files. This is an experimental evaluation workflow; it does not
promote a model or establish release qualification.

Keep queues, ledgers, packs, reports and commitments in an access-controlled
private location. The queue contains plaintext. Reports contain hashes, counts
and aggregate metrics but no request text or per-request predictions; hashes
and counts can still be sensitive. Local hashes and reviewer IDs are not
signatures or proof of source rights, reviewer identity or independence.

1. The holdout custodian selects already reviewed test-only queues and ledgers
   and creates a new commitment without running the candidate. The command
   reads those files, requires decisions and test-only approvals, and writes
   their digests rather than their text or labels. Repeat `--holdout-input` for
   each source. Record the resulting output digest independently before the
   candidate is frozen and keep the test files unavailable to its operator.

```powershell
python ml/lfg-encoder/evaluate_embedded_holdout.py commit-holdout `
  --holdout-input PRIVATE_TEST_QUEUE.jsonl PRIVATE_TEST_LEDGER.json `
  --snapshot SNAPSHOT.json --node NODE_ID --output NEW_PRIVATE_COMMITMENT.json
```

2. Freeze using only development queues and the commitment. The policy supplied
   to `--policy` must be the train/validation-only v2 policy; `--final-policy`
   is the separately chosen v1 policy that sets train, validation and test
   support requirements for the later evaluation. The command validates the
   pack against the development audit, invokes the external and embedded
   candidates on validation data, requires matching choices for every item and
   at least one escalation, and records the selected binary, model, pack,
   settings, policies and evaluator identities. The referenced external files
   provide an audit and identity check even though inference in the binary is
   embedded. Select an already built `embedded-node` executable.

```powershell
python ml/lfg-encoder/evaluate_embedded_holdout.py freeze `
  --input PRIVATE_DEV_QUEUE.jsonl PRIVATE_DEV_LEDGER.json `
  --snapshot SNAPSHOT.json --node NODE_ID --policy DEV_POLICY.json --final-policy FINAL_POLICY.json `
  --binary EMBEDDED_LFG_BINARY --artifact ENCODER_ARTIFACT_DIR `
  --manifest-sha256 TRUSTED_MANIFEST_SHA256 --runtime-config RUNTIME.json `
  --pack PRIVATE_PACK.json --pack-sha256 TRUSTED_PACK_SHA256 --exemplar-config EXEMPLAR_CONFIG.json `
  --commitment PRIVATE_COMMITMENT.json --commitment-sha256 TRUSTED_COMMITMENT_SHA256 `
  --max-items 1000 --max-input-bytes 10485760 --max-output-bytes 16000000 `
  --timeout-seconds 180 --output NEW_PRIVATE_FREEZE.json
```

Repeat `--input` for each development queue/ledger. The limits shown are
examples; choose them for the actual corpus. The output byte limit is checked
after process output is captured in memory, so it is not a memory limit against
an untrusted executable. The native command's own configured output limit also
applies. Only use trusted local binaries and files.

3. Independently record the freeze file's digest before giving the evaluator
   access to the test files. Run `test` with the same arguments used for `freeze`,
   a different new `--output`, and these additional arguments:

```powershell
  --freeze PRIVATE_FREEZE.json --freeze-sha256 TRUSTED_FREEZE_SHA256 `
  --holdout-input PRIVATE_TEST_QUEUE.jsonl PRIVATE_TEST_LEDGER.json
```

The final command rechecks the frozen development candidate before opening test
queues, verifies that the supplied holdout files match the precommitment, and
audits combined project, group, family, normalized-text and request-ID separation
and configured support. It then runs the deterministic kernel and the embedded
candidate on the reviewed test records. The embedded result must carry the frozen
identities and exactly match the separately run kernel result; the candidate is
scored only on target-node escalations. At least one such escalation is required.
Inputs and the frozen candidate are checked again after evaluation. The report
is aggregate and marked `qualificationEstablished: false`.

The commitment does not prove when it was created or prevent someone with access
to the holdout from inspecting it, selecting a favorable set, making another
commitment or running the test repeatedly. Keep roles and files separate, record
digests in an independently trusted record, and enforce any one-use policy
outside this tool. The audit cannot detect all related projects, paraphrases,
prior exposure, incorrect review or source-rights issues. The available test
examples are selected from kernel observations, which may be biased. The
reported conditional choice metrics do not establish end-to-end usefulness,
confidence calibration, latency or release readiness.

## Local test

On 2026-10-31, the complete commitment, freeze and test sequence ran against the
actual local encoder and embedded executable using **simulated test-only review
attestations** and six test-only strings. The final runner observed the candidate
process and produced a two-item aggregate fixture report marked unqualified.
No real project requests or labels were used in that smoke test. The actual
11-item node queue remains pending with zero approved decisions; no real
holdout quality report has been produced.
