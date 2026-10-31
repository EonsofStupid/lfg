# Disposable checked-merge runner

The checked-merge workflow can use a repository-configured runner for the
explicitly labeled `codex/` pull requests it is allowed to merge. Without the
`LFG_CHECKED_MERGE_RUNNER` repository variable it continues to use
`ubuntu-latest`. The ordinary CI, release and SBOM workflows remain on their
existing runners; configuring this runner does not make those checks pass.

`ci/isolated-runner` contains an image and a Windows PowerShell supervisor for
one-job, repository-scoped just-in-time runners. The supervisor obtains its
short-lived configuration through the locally authenticated GitHub CLI, sends
it over container stdin, and starts another runner after a job finishes. The
runner image starts as an unprivileged user; the example invocation applies
CPU, memory and process limits, drops Linux capabilities, enables
`no-new-privileges`, and supplies no host volumes or Docker socket. The image
base is pinned by digest and can be changed with the Docker build argument;
the packages installed from Ubuntu repositories are not version-locked, so the
image is not a reproducible release artifact.

Build from the isolated context, then copy the example configuration to a
private, operator-selected path and fill in the repository and resource limits:

```powershell
docker build -t lfg-isolated-actions-runner:local ci/isolated-runner
powershell -NoProfile -File ci/isolated-runner/supervise.ps1 -Config PRIVATE_CONFIG.json
```

The host must have Docker Desktop running and a GitHub CLI login with repository
administration access. Set the repository variable to a JSON runner selection,
for example `["lfg-checked-merge"]`, and choose the same label in
the supervisor configuration. The checked-merge workflow selects that runner
only for a same-repository, non-draft `codex/` PR marked
`ready-to-auto-merge`. It tests the checkout and, after verification, checks
that both head and base still match before requesting the merge. Branch rules
and GitHub permissions still apply. A changed head or base requires a new run.
The kernel golden comparison remains mandatory and logs both file digests.
Uploading its output is best-effort because an exhausted GitHub artifact quota
otherwise prevents the remaining verification from running; the upload failure
is visible in the workflow and the output file is unavailable after the
disposable container exits. Restore artifact capacity to retain that file in
GitHub. The printed digest and successful comparison are not a retained copy.

The supervisor registers only the selected custom label, not the generic
`self-hosted` label, so jobs requesting only the generic label cannot claim
this runner. The custom label is still a scheduler selector, **not an authorization boundary**. Other
workflows able to target it can run code on this runner; repository and fork
workflow permissions must be appropriate for that risk. Containers share the
host kernel and retain network access, including potentially to local services;
they are not a hardened VM or a complete sandbox. Do not mount private review
data, credentials, host directories or the Docker socket into them. Job output
and supervisor logs may contain sensitive information and must be protected.
The host GitHub CLI credential is not mounted into the container, but the job
receives whatever GitHub token and secrets its workflow grants. Limit those
permissions independently.

The supervisor runs only while its host process and Docker Desktop remain up;
it is not installed as a service and does not restart after reboot. Stopping it
may leave an active job container or an offline runner registration; inspect
the exact `lfg-jit-...` container and repository runner before stopping or
removing anything. Do not delete unrelated containers or runners. A runner can
be deregistered only after confirming that its job has finished and the runner
belongs to this supervisor. Review and training workspaces are separate and are
not needed by this CI path.
