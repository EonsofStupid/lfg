param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^v\d+\.\d+\.\d+([\-+].*)?$")]
    [string]$VersionTag,
    [string]$Message = "",
    [switch]$SkipChecks,
    [switch]$AcknowledgeChecklist
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [scriptblock]$Command,
        [string]$FailureMessage
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code $LASTEXITCODE)"
    }
}

if (-not $Message) {
    $Message = "Release $VersionTag"
}

if (-not $AcknowledgeChecklist) {
    throw "Review .github/RELEASE_CHECKLIST.md and rerun with -AcknowledgeChecklist."
}

$gpg = Get-Command gpg -ErrorAction SilentlyContinue
if ($null -eq $gpg) {
    throw "gpg is required for signed tags. Install GnuPG and configure a signing key."
}

Invoke-NativeChecked -Command { git rev-parse --is-inside-work-tree | Out-Null } -FailureMessage "not inside a git repository"
$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "failed to resolve repository root"
}

if (-not $SkipChecks) {
    Write-Host "Running release preflight checks..." -ForegroundColor Cyan
    Invoke-NativeChecked -Command {
        & (Join-Path $repoRoot "scripts\pre-release-check.ps1")
    } -FailureMessage "release preflight checks failed"
}

$tagExists = (git tag --list $VersionTag)
if ($tagExists) {
    throw "tag $VersionTag already exists"
}

Invoke-NativeChecked -Command { git tag -s $VersionTag -m $Message } -FailureMessage "failed to create signed tag"
Invoke-NativeChecked -Command { git tag -v $VersionTag } -FailureMessage "failed to verify signed tag"

Write-Host "Signed tag created and verified: $VersionTag" -ForegroundColor Green
Write-Host "Push with: git push origin $VersionTag"
