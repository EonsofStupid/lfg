# Checked merges

The `Verify and merge completed work` workflow runs the UI tests and build,
Rust formatting and workspace tests (including embedded assets), an optional
native ONNX compilation check, and the Python tooling tests for pull requests.
It does not claim model quality, release qualification, or real-data coverage.

Once a pull request has been reviewed and its work is complete, the working
agent applies the `ready-to-auto-merge` label to start verification. If it passes, the workflow
merges only a non-draft branch in this repository targeting `main`, regardless
of branch naming convention.
It checks that the live pull-request head and base still match the tested
revisions and asks GitHub to reject a changed head. It does not merge other
branches or forked contributions. A conflict or failed check leaves the pull
request open. Before merging, it waits for the latest matching `CI` run for the
same pull request, head and base, and requires successful `rust`, `python-ml`,
`ui` and `container` jobs from that run's current attempt. A missing, skipped,
failed or cancelled required job cannot satisfy this gate. A failed CI run or
timeout leaves the PR open; after fixing or rerunning CI, rerun checked merge
or reapply the label. If the base changes, update the branch or rerun verification.

Agents must not use `gh pr merge` directly, including `--auto`. On this
repository GitHub can merge immediately without waiting for CI. Use the
`ready-to-auto-merge` checked workflow even when GitHub permits a direct merge.

The private repository currently has no branch protection: this workflow does
not prevent another actor from merging directly, and the base check and merge
are separate GitHub operations, leaving a small race if `main` changes between
them. GitHub branch protection with required checks or a merge queue is needed
to enforce those guarantees. Never apply the label to unfinished work. The
workflow is not a substitute for the pre-release qualification steps.

The earlier live path merged pull requests, including #110 on 2026-11-01 using
the configured isolated runner. The full-CI wait is a later addition; its live
merge behavior must also be verified. Other workflows can still report failures
outside this gate. If no suitable runner is available, verification and
automatic merging cannot proceed.

The separate container CI job has exhausted disk on GitHub-hosted runners
while building the image. Before that build it now reports disk space and
removes three unused SDK directories from the disposable hosted runner only.
This leaves the image's dependencies and security checks intact; it does not
prove that the available space is sufficient or qualify the image. A
non-hosted runner skips this cleanup.
The SBOM baseline review image job uses the same guarded cleanup script.
