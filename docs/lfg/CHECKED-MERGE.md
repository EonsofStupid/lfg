# Checked merges

The `Verify and merge completed work` workflow runs the UI tests and build,
Rust formatting and workspace tests (including embedded assets), an optional
native ONNX compilation check, and the Python tooling tests for pull requests.
It does not claim model quality, release qualification, or real-data coverage.

Once a pull request has been reviewed and its work is complete, applying the
`ready-to-auto-merge` label starts verification. If it passes, the workflow
merges only a non-draft `codex/` branch in this repository targeting `main`.
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
