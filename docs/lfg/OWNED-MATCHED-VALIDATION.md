# Matched owned and deterministic validation

**Status: blocked before private inference.** The original scope-and-pipe
transport failed independent review and `_ssh_worker` refuses to run it. A
replacement file handshake and zero-row probe are implemented. A specific run
requires review and a successful target-host probe for the current code and
configuration. Local tests alone do not establish isolation or cleanup.

The intended evaluation compares the owned assessment route and deterministic
rule hits with the previously recorded classical and local 12B results on the
same screened development validation. It does not train, distill, open heldout
material, or qualify or promote a model. Labels and prior validation-use claims
retain the review status recorded by the baseline and are not independently
verified here.

## Baseline binding

The local binder requires the classical report and config and the 12B report
and config by path and SHA-256. It reloads only declared train and validation
sources, reruns screening and source identity checks, applies the review ledger
and rejected-label policy, and checks row IDs, task labels, known-target masks,
and support counts. It reconstructs the 12B worker's exact newline-terminated
input and compares its digest with the prior report. This binds the screened
texts and masks to the report's recorded input bytes. The report's field named
`response_record_id_set_sha256` is compared too, but the prior report assembler
computed that field from the expected IDs; it is not independent proof that
each row produced an answer.

The 12B report contains aggregate predictions, not per-row predictions. A
paired intersection with owned predictions cannot be reconstructed from it.
Different denominators and unknown predictions must remain explicit.

## Interpretation

The owned output contract has a free-form single intent rather than a proven
complete multilabel taxonomy. The existing checkpoint framing does not establish
that an exact target literal has the same meaning as the bound target, so owned
predictions remain unknown. A different intent, invalid output, timeout, missing
result, and other incomplete response are never negatives. Metrics for an
unknown-only result have zero coverage and null precision, recall, and F1.

The deterministic endpoint can yield a configured rule-hit outcome only when
its actual configuration identity, complete hit list, and untruncated scoring
are verified. That is deterministic rule behavior, not learned-model quality.

## File protocol and next checks

The new path creates a unique private workspace and detached transient user
service. A fixed root bootstrap creates a network namespace, enables loopback,
then drops privilege before starting user-owned code. The worker checks its
identity, capabilities, namespace and routes, starts a fresh authenticated
service, checks its identity and status, and publishes a nonce-bound marker.
A separate host-side coordinator checks the unit, process identities, cgroup,
namespace and host loopback before transferring a payload. It verifies the
transferred bytes and promotes an exclusive file for the waiting worker. The
worker returns aggregate-only results. The coordinator stops only its unique
unit, checks the observed cgroup and namespace, compares coexistence snapshots,
and removes the workspace only when cleanup is verified. A transfer timeout is
reported as uncertain. Same-owner or root access to private files and processes
is outside this isolation boundary.

The bootstrap passes only numeric owner IDs and file paths as positional data;
it runs no user-owned code until after `setpriv`. The pre-existing sudo grant
for `unshare`, if broad, can independently authorize other root commands; this
program's fixed bootstrap does not constrain that grant. The report records the
systemd MainPID identity separately: sudo may remain privileged while the worker
and server are verified as unprivileged.

The operator supplies resource floors, CPU and memory limits, timeouts, and
artifact identities in the private configuration. The file protocol also
requires `file_preflight_deadline_seconds`, `file_transfer_deadline_seconds`,
`file_cleanup_deadline_seconds`, `file_max_binary_bytes`,
`file_max_metadata_bytes`, `file_min_gpu_free_bytes`, and
`expected_remote_architecture`. `service_config_template_sha256` pins the
source template; the runtime config digest is reported separately because the
isolated metadata paths differ. A zero GPU floor
means no GPU free-memory threshold is enforced; unavailable GPU telemetry is
reported as unknown. Select limits from current coexistence measurements.

Before private inference, an independent reviewer must accept the implementation
and the actual production protocol must pass a zero-row probe on the target
host. Coordinate resource use first. The probe uses the same worker and
coordinator, starts the pinned service and transfers an empty envelope; it
does not load corpus rows or submit inference requests:

```powershell
python ml/lfg-encoder/compare_owned_validation.py --file-probe `
  --config <owner-private-config.json> --output <new-private-probe-directory>
```

The subsequent run uses `--probe-report <private-probe-directory>/owned-file-probe.json`.
The adapter checks that the successful probe matches the current adapter,
generated worker, control code, and configuration digests. A code or config
change requires a new probe. The report does not itself establish independent
review; record that evidence separately. Do not use the old scope launcher.

Local regression tests cover mapping, baseline binding, file gates and bounded
process behavior. Some historical tests cover the rejected generated code.
Passing them does not verify the target host, the model, or the eleven-row
evaluation.
