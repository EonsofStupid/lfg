param(
    [Parameter(Mandatory = $true)][string]$Commit,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateRange(1, 256)][int]$Jobs = 2
)

$ErrorActionPreference = 'Stop'
if ($Commit -cnotmatch '^[0-9a-f]{40}$') {
    throw 'Commit must be a full lowercase commit ID.'
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $destination) {
    throw 'Output directory already exists; select a new directory.'
}
$environmentNames = @('CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'CARGO_ENCODED_RUSTFLAGS',
    'RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'LFG_UI_DIST')
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Invoke-Checked([scriptblock]$Command, [string]$Message) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Message (exit $LASTEXITCODE)" }
}

function Get-AssetHashes([string]$Root) {
    $hashes = [ordered]@{}
    $prefix = $Root.TrimEnd('\') + '\'
    foreach ($file in (Get-ChildItem -LiteralPath $Root -File -Recurse | Sort-Object FullName)) {
        $relative = $file.FullName.Substring($prefix.Length).Replace('\', '/')
        $hashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if (-not $hashes.Contains('index.html')) { throw 'UI build did not produce index.html.' }
    return $hashes
}

Push-Location $repo
try {
    $resolved = (& git rev-parse --verify "$Commit^{commit}").Trim()
    if ($LASTEXITCODE -ne 0 -or $resolved -cne $Commit) { throw 'Commit could not be verified.' }
    New-Item -ItemType Directory -Path $destination -ErrorAction Stop | Out-Null
    $archive = Join-Path $destination 'source.tar'
    Invoke-Checked { git archive --format=tar "--output=$archive" $Commit } 'Source archive failed'
    $outputs = @()
    foreach ($suffix in @('a', 'b')) {
        $source = Join-Path $destination "source-$suffix"
        $target = Join-Path $destination "target-$suffix"
        New-Item -ItemType Directory -Path $source -ErrorAction Stop | Out-Null
        Invoke-Checked { tar -xf $archive -C $source } 'Source extraction failed'
        $ui = Join-Path $source 'ui\master-clyffy-web'
        Push-Location $ui
        try {
            Invoke-Checked { npm ci --offline --no-audit --no-fund } 'Offline UI installation failed'
            Invoke-Checked { npm run build } 'UI build failed'
        } finally { Pop-Location }
        $assets = Join-Path $ui 'dist'
        $env:CARGO_TARGET_DIR = $target
        $env:CARGO_BUILD_JOBS = [string]$Jobs
        $env:RUSTFLAGS = $null
        $env:CARGO_ENCODED_RUSTFLAGS = @('-C', 'target-feature=+crt-static', '-C', 'link-arg=/Brepro',
            "--remap-path-prefix=$source=/lfg-src", "--remap-path-prefix=$target=/lfg-target") -join [char]31
        $env:RUSTC_WRAPPER = $null
        $env:RUSTC_WORKSPACE_WRAPPER = $null
        $env:LFG_UI_DIST = $assets
        Push-Location $source
        try {
            Invoke-Checked { cargo build --offline --locked --release -p lfg-server --features embedded-ui --target x86_64-pc-windows-msvc } 'Cargo build failed'
        } finally { Pop-Location }
        $binary = Join-Path $target 'x86_64-pc-windows-msvc\release\lfg-server.exe'
        $outputs += [ordered]@{
            binary = $binary
            sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $binary).Length
            assets = (Get-AssetHashes $assets)
        }
    }
    $assetsEqual = (ConvertTo-Json $outputs[0].assets -Compress) -ceq (ConvertTo-Json $outputs[1].assets -Compress)
    $report = [ordered]@{
        schemaVersion = 'lfg.windows-baseline-build.v1'
        commit = $Commit
        archiveSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        rustc = (& rustc --version).Trim()
        cargo = (& cargo --version).Trim()
        node = (& node --version).Trim()
        npm = (& npm --version).Trim()
        jobs = $Jobs
        assetsIdentical = $assetsEqual
        binariesIdentical = ($outputs[0].sha256 -ceq $outputs[1].sha256)
        outputs = $outputs
        scope = 'Same-machine build comparison; not model or release qualification.'
    }
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $destination 'build-report.json') -Encoding UTF8
    if (-not $report.assetsIdentical -or -not $report.binariesIdentical) {
        throw 'Build outputs differ; inspect the retained report and artifacts.'
    }
    $report | ConvertTo-Json -Depth 8
} finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
    }
    Pop-Location
}
