# GB10 training smoke evidence, 2026-10-30

This is infrastructure evidence, not a quality, performance, security or
promotion qualification. The checked-in six-example synthetic dataset and
`--allow-below-gate` were used intentionally. The randomly initialized tiny
model is not suitable for product inference.

- Host: Linux aarch64, NVIDIA GB10; NVIDIA driver `580.178.04`. A separate
  SWAG container remained running throughout the smoke test.
- Image: existing NVIDIA vLLM image with local image ID
  `sha256:654e563e727be1968487d453de0733fb074cb59adf424c779d0f9e7cfcb2b6b6`.
  A separate venv inherited system packages and installed Transformers 4.57.6,
  datasets 4.8.5 and accelerate 1.15.0. Pip reported conflicts with the base
  image's vLLM dependencies; this environment is a test, not a qualified or
  dependency-locked training image. PyTorch was `2.12.0a0+5aff3928d8.nv26.5.50603568`,
  CUDA runtime `13.2`, Python `3.12.3`.
- Model: `hf-internal-testing/tiny-random-bert`, immutable revision
  `f171d7baecaf37b5da5a3616d8833b9969753535`. Mismatched weight sizes were
  explicitly allowed for this smoke test; loading reported a newly initialized
  classification head.
- Dataset SHA-256: `93640be08ad6da46e6124e99753fc349efb60cd43b967f55c4fff83695b1724e`.
  Trainer SHA-256: `054148eed1985ba80cb4ec0bb7009d7149aae7282a2b62b88b7ed735a379c438`.
  Run manifest records configuration, policy, split and dependency identities.
- Limits: separate Docker container capped at 6 GiB and two CPUs; no changes to
  the running service. GPU is shared, not isolated by those limits.
- A test completed on GB10, saved artifacts and checkpoints, and a fresh
  network-disabled container reloaded the model for finite CUDA inference.
  Its optimizer state included CUDA tensors.
- A second run was deliberately killed after checkpoint 10, exit 137. After
  verifying the container had stopped, its stale lock was removed. A new
  container with the same recorded run inputs resumed and finished at step 450,
  exit 0. A fresh network-disabled process reloaded the resulting model on
  `cuda:0`, produced finite `[1, 13]` logits and read CUDA optimizer state.

The artifact and run outputs remain in the operator's experiment directory on
the Spark. The short-lived tests are not continuously running, are not an
automatic training service, and do not establish reproducibility across hosts
or guarantee recovery from every failure. A production lane still needs a
reviewed and separated corpus, exact dependency lock and image provenance,
resource isolation and supervision, quality evaluation, export and CPU
qualification, and a governed promotion path.
