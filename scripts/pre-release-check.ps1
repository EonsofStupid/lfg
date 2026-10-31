param(
    [string]$ImageName = "lfg:release-candidate",
    [string]$OutputPath = "output\\release\\preflight.json",
    [switch]$SkipContainerBuild
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

function Add-CheckResult {
    param(
        [System.Collections.Generic.List[object]]$Results,
        [string]$Name,
        [string]$Status,
        [string]$Detail
    )
    $Results.Add([pscustomobject]@{
        name = $Name
        status = $Status
        detail = $Detail
    }) | Out-Null
}

$results = New-Object 'System.Collections.Generic.List[object]'
$startedAt = [DateTimeOffset]::UtcNow

Invoke-NativeChecked -Command { git rev-parse --is-inside-work-tree | Out-Null } -FailureMessage "not inside a git repository"
$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "failed to resolve repository root"
}
Set-Location $repoRoot

$checklistPath = Join-Path $repoRoot ".github\RELEASE_CHECKLIST.md"
if (-not (Test-Path $checklistPath)) {
    throw "release checklist is missing at $checklistPath"
}
Add-CheckResult -Results $results -Name "release-checklist-present" -Status "passed" -Detail $checklistPath

$statusLines = git status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "git status failed"
}
if ($statusLines) {
    throw "working tree is not clean; commit or stash changes before tagging"
}
Add-CheckResult -Results $results -Name "working-tree-clean" -Status "passed" -Detail "no pending git changes"

Invoke-NativeChecked -Command { cargo check --workspace --locked } -FailureMessage "cargo check failed"
Add-CheckResult -Results $results -Name "cargo-check" -Status "passed" -Detail "cargo check --workspace --locked"

Invoke-NativeChecked -Command { cargo test --workspace --locked } -FailureMessage "cargo test failed"
Add-CheckResult -Results $results -Name "cargo-test" -Status "passed" -Detail "cargo test --workspace --locked"

Invoke-NativeChecked -Command { python -m py_compile "ml/lfg-encoder/train.py" "ml/lfg-encoder/distill.py" } -FailureMessage "python py_compile failed"
Add-CheckResult -Results $results -Name "python-py-compile" -Status "passed" -Detail "train.py + distill.py syntax"

Invoke-NativeChecked -Command { python "ml/lfg-encoder/train.py" validate } -FailureMessage "train.py validate failed"
Add-CheckResult -Results $results -Name "dataset-validate" -Status "passed" -Detail "synthetic dataset gate"

Invoke-NativeChecked -Command { python "ml/lfg-encoder/distill.py" --help } -FailureMessage "distill.py --help failed"
Add-CheckResult -Results $results -Name "distill-cli" -Status "passed" -Detail "distill cli available"

Push-Location "ui/master-clyffy-web"
try {
    Invoke-NativeChecked -Command { npm ci } -FailureMessage "npm ci failed"
    Invoke-NativeChecked -Command { npm run build } -FailureMessage "npm run build failed"
} finally {
    Pop-Location
}
Add-CheckResult -Results $results -Name "ui-build" -Status "passed" -Detail "npm ci && npm run build"

if (-not $SkipContainerBuild) {
    Invoke-NativeChecked -Command { .\scripts\deploy.ps1 -Action BuildImage -ImageName $ImageName -BuildVersion "preflight" } -FailureMessage "container build preflight failed"
    Add-CheckResult -Results $results -Name "container-build" -Status "passed" -Detail "deploy BuildImage"
}

$sbomJsonPath = "output/sbom/lfg-policy.json"
Invoke-NativeChecked -Command {
    .\scripts\generate-sbom.ps1 -ImageName $ImageName -OutputPath $sbomJsonPath -Format json
} -FailureMessage "SBOM generation for policy failed"
Invoke-NativeChecked -Command {
    python "scripts/validate-sbom-policy.py" --sbom $sbomJsonPath --policy "security/sbom-policy.json"
} -FailureMessage "SBOM policy validation failed"
Add-CheckResult -Results $results -Name "sbom-policy" -Status "passed" -Detail "validate-sbom-policy.py gate"

$completedAt = [DateTimeOffset]::UtcNow
$report = [pscustomobject]@{
    schemaVersion = "lfg.release-preflight.v1"
    startedAt = $startedAt.ToString("o")
    completedAt = $completedAt.ToString("o")
    durationSeconds = [math]::Round(($completedAt - $startedAt).TotalSeconds, 2)
    skipContainerBuild = [bool]$SkipContainerBuild
    checks = $results
}

$outputAbsolute = Resolve-Path "." | ForEach-Object { Join-Path $_.Path $OutputPath }
$outputDirectory = Split-Path -Parent $outputAbsolute
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -Path $outputAbsolute -Encoding UTF8

Write-Host "Release preflight checks passed." -ForegroundColor Green
Write-Host "Report: $outputAbsolute"
