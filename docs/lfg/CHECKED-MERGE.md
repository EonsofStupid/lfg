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
request open. If the base changes, update the branch or rerun verification.

The private repository currently has no branch protection: this workflow does
not prevent another actor from merging directly, and the base check and merge
are separate GitHub operations, leaving a small race if `main` changes between
them. GitHub branch protection with required checks or a merge queue is needed
to enforce those guarantees. Never apply the label to unfinished work. The
workflow is not a substitute for the pre-release qualification steps.

The live path has merged pull requests, including #110 on 2026-11-01 using the
configured isolated runner. The separate CI workflows can still report failures
that are not covered by this workflow's merge condition; inspect them and resolve
release-relevant failures before a pre-release. If no suitable runner is
available, verification and automatic merging cannot proceed.

The separate container CI job has exhausted disk on GitHub-hosted runners
while building the image. Before that build it now reports disk space and
removes three unused SDK directories from the disposable hosted runner only.
This leaves the image's dependencies and security checks intact; it does not
prove that the available space is sufficient or qualify the image. A
non-hosted runner skips this cleanup.
The SBOM baseline review image job uses the same guarded cleanup script.
