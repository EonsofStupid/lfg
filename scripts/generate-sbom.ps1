param(
    [string]$ImageName = "lfg:latest",
    [string]$OutputPath = "output\sbom\lfg.spdx",
    [ValidateSet("spdx", "spdx-json", "json")]
    [string]$Format = "spdx"
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

$outputAbsolute = Resolve-Path "." | ForEach-Object { Join-Path $_.Path $OutputPath }
$outputDirectory = Split-Path -Parent $outputAbsolute
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($null -eq $docker) {
    throw "docker was not found on PATH"
}

$syft = Get-Command syft -ErrorAction SilentlyContinue
if ($syft) {
    Write-Host "Generating SBOM with syft..." -ForegroundColor Cyan
    Invoke-NativeChecked -Command { syft $ImageName -o "$Format=$outputAbsolute" } -FailureMessage "syft SBOM generation failed"
    Write-Host "SBOM written to $outputAbsolute" -ForegroundColor Green
    return
}

$null = & docker scout version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Neither syft nor docker scout is available; install syft or Docker Scout."
}

Write-Host "Generating SBOM with docker scout..." -ForegroundColor Cyan
$scoutFormat = if ($Format -eq "spdx-json") { "json" } else { $Format }
Invoke-NativeChecked -Command { docker scout sbom --format $scoutFormat --output $outputAbsolute $ImageName } -FailureMessage "docker scout SBOM generation failed"
Write-Host "SBOM written to $outputAbsolute (format: $scoutFormat)" -ForegroundColor Green
