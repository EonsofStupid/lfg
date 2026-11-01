# Local Windows CUDA assessment lab

**Historical fixture environment:** The lock and environment described below
include `fsspec 2026.2.0` and `transformers 4.57.6`, which a later container
scan flagged for high-severity vulnerabilities. Do not use that environment
for real training or untrusted inputs. It is retained only to identify the
fixture experiment; use a separately verified patched environment instead.

On 2026-10-31 the local machine reported an NVIDIA GeForce RTX 5090 Laptop GPU
with compute capability 12.0, 24,463 MiB total memory and driver 592.01. The
existing Python environment had a CPU-only Torch build. A new isolated Python
3.11 environment under the ignored `target/lfg-win-cu130-lab-20261031-v1`
directory was installed from a 54-wheel local wheelhouse with `--no-index` and
`--require-hashes`. Its [wheel lock](../../ml/lfg-encoder/locks/windows-cuda-cu130-py311-experimental.requirements.txt)
pins Torch `2.14.1+cu130` and the exact selected dependencies. The original CPU
wheelhouse matched its committed lock; its non-Torch wheels were copied, and
the CUDA Torch wheel was downloaded from the PyTorch CUDA 13.0 index.

`pip check` passed. The installed-wheel payload verifier checked 25,332 files
and 3,444,306,759 bytes against the wheelhouse; wheel records and three scripts
were excluded. This does not authenticate the publisher, interpreter, bootstrap
pip, driver or excluded files. PyTorch describes Blackwell support in its
[2.7 release notes](https://pytorch.org/blog/pytorch-2-7/); actual compatibility
still needs checking on each machine and workload.

A later isolated candidate environment uses `datasets 5.1.0`, `fsspec
2026.7.0` and `transformers 5.22.1`, outside the affected ranges for the four
findings reported by the container scan. The [candidate lock](../../ml/lfg-encoder/locks/windows-cuda-cu130-py311-20261031.requirements.txt)
and [fixture probe report](evidence/windows-5090-assessment-cuda-candidate.v1.json)
identify the tested inputs. `pip check` passed and the probe verified 28,091
installed wheel-payload files before fitting the four fixture heads on the
same GPU. This is a compatibility check, not a complete security assessment,
real training, or evidence that the ARM64 Spark environment has been updated.
The historical lock above remains unchanged so its earlier result is traceable.
See [the dependency update](ML-DEPENDENCY-SECURITY.md) for the scan findings
and the limits of the candidate check.

A tiny generated tensor produced finite CUDA gradients and a parameter update.
The assessment encoder and all four ridge heads then ran on CUDA with a
locally pinned MiniLM snapshot, generated fixture texts and labels, and a 3%
PyTorch allocator fraction. The first direct probe reported 124,605,440 bytes
peak allocated and 142,606,336 bytes peak reserved. The loaded model is a
fixture for testing this path, not LFG's own model or an adopted product model.
No private requests were read, no weights were saved, and this is not a real
training run, accuracy result or optimized-binary qualification. The machine
was shared with other workloads; a single elapsed time is not a benchmark, and
the allocator fraction does not isolate the GPU from other processes.

A subsequent run of the checked-in probe produced this [aggregate report](evidence/windows-5090-assessment-cuda-fixture.v1.json)
with SHA-256 `97c9d0025de6c4ac2b313a3218fd2209c7e59ae19a70ec9f561bbb8a3994939c`.
It fit all four heads and reported 124,754,944 bytes peak allocated and
142,606,336 bytes peak reserved. Its recorded source and configuration hashes
refer to the Windows working-tree bytes; some of those files used CRLF line
endings and differ byte-for-byte from the LF Git blobs. The [fixture model
manifest](../../ml/lfg-encoder/locks/minilm-l6-v2.1110a243.fixture.manifest.json)
records the selected snapshot hashes, not an independent publisher or license
attestation. This was one run, not proof of numerical repeatability.

`probe_assessment_cuda.py` provides a repeatable fixture check with explicit
inputs and device/resource selection. It verifies the model snapshot and the
locked installed-wheel payload before the probe, requests deterministic Torch
operations, checks source identities and version fingerprints again afterward,
and writes an aggregate report without learned weights. The payload itself is
not reverified after execution. The script uses fixed fixture strings only; it
does not accept a real corpus. For example, after preparing the local snapshot,
manifest, wheelhouse and environment:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& PRIVATE_VENV\Scripts\python.exe ml/lfg-encoder/probe_assessment_cuda.py `
  --model LOCAL_MODEL_DIRECTORY --model-manifest LOCAL_MODEL_MANIFEST.json `
  --revision PINNED_MODEL_REVISION --config OPERATOR_CONFIG.json `
  --environment-lock LOCKED_REQUIREMENTS.txt --wheelhouse LOCAL_WHEELHOUSE `
  --allowed-extra pip --device-index 0 --cuda-memory-fraction 0.03 `
  --output NEW_PRIVATE_REPORT.json
```

The config's device is explicitly overridden to CUDA for this probe; its batch,
thread and size limits remain effective. Select an allocator fraction that is
appropriate for the user's machine and other processes. `CUBLAS_WORKSPACE_CONFIG`
is an operator-selected setting, not a guarantee of identical results across
software or hardware. The environment and model identities, device, resource
limit and source digests are recorded in the report.

The second Spark, `clyffy-03`, was online during this work but rejected SSH as
the current Windows user under its tailnet policy. The local GPU lab does not
establish that Spark is usable or replace the planned Spark training run.
