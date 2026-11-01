# ML dependency update

The container scan in [CI run 41263221608](https://github.com/EonsofStupid/lfg/actions/runs/41263221608)
reported four high-severity findings in `fsspec 2026.2.0` and `transformers
4.57.6`. The reported affected ranges and fixes are documented in the GitHub
advisories for [fsspec CVE-2026-104851](https://github.com/advisories/GHSA-27vj-qcqg-25rc)
and Transformers [CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72),
[CVE-2026-5241](https://github.com/advisories/GHSA-fgcw-684q-jj6r), and
[CVE-2026-9856](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w). A
versioned model manifest and `trust_remote_code=False` are useful controls but
are not a substitute for fixing vulnerable dependencies.

The default requirements now select `datasets >=5.1,<6`, `transformers
>=5.22.1,<6` and `fsspec >=2026.6`. The dataset library constrains compatible
`fsspec` versions during resolution. These are compatibility ranges; a
release or training run still needs an exact wheel lock and an up-to-date
security scan. Do not change historical locks in place: they identify prior
experiments but include the flagged versions and must not be used for real
training or untrusted inputs.

An isolated Windows CUDA candidate resolved `datasets 5.1.0`, `transformers
5.22.1`, and `fsspec 2026.7.0`; its exact wheel hashes and fixture GPU report
are linked from [the local CUDA lab](LOCAL-CUDA-LAB.md). `pip check`, the full
Python suite with the local tokenizer and trainer scheduling tests enabled, and
the bounded CUDA assessment-head fixture probe passed. This does not establish
that the Linux image scan is clean, that the Spark environment is updated, or
that a model trained on user data is qualified.

After the updated image passed the high/critical scan, its SBOM contained one
`oci` entry for the image itself. Syft identified it as the SPDX document's
single described container; the existing package-type policy rejected it.
The policy can now configure permitted namespaces for that exact document root.
It still counts the root in package growth and treats any other OCI package,
ambiguous root, or unconfigured root namespace as an ordinary package subject
to the existing type policy. The policy check passed locally on the actual CI
SBOM; a fresh CI run is still required for the full image and smoke test.
