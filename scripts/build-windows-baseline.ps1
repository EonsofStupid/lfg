param(
    [Parameter(Mandatory = $true)][string]$Commit,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateRange(1, 256)][int]$Jobs = 2,
    [string]$ModelArtifactDir,
    [string]$ModelManifestSha256
)

$ErrorActionPreference = 'Stop'
if ($Commit -cnotmatch '^[0-9a-f]{40}$') {
    throw 'Commit must be a full lowercase commit ID.'
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$builderSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
if (Test-Path -LiteralPath $destination) {
    throw 'Output directory already exists; select a new directory.'
}
$hasModelDirectory = -not [string]::IsNullOrWhiteSpace($ModelArtifactDir)
$hasModelDigest = -not [string]::IsNullOrWhiteSpace($ModelManifestSha256)
if (($hasModelDirectory -ne $hasModelDigest) -or
    (($PSBoundParameters.ContainsKey('ModelArtifactDir') -or $PSBoundParameters.ContainsKey('ModelManifestSha256')) -and
     (-not $hasModelDirectory -or -not $hasModelDigest))) {
    throw 'ModelArtifactDir and ModelManifestSha256 must be supplied together.'
}
if ($hasModelDigest -and $ModelManifestSha256 -cnotmatch '^[0-9a-f]{64}$') {
    throw 'ModelManifestSha256 must be a lowercase SHA-256 digest.'
}
$environmentNames = @('CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS', 'CARGO_ENCODED_RUSTFLAGS',
    'RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'LFG_UI_DIST',
    'LFG_EMBEDDED_ONNX_DIR', 'LFG_EMBEDDED_ONNX_SHA256')
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

function Get-ModelHashes([string]$Root, [string]$ExpectedManifest) {
    $directory = Get-Item -LiteralPath $Root -Force
    if (-not $directory.PSIsContainer -or ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Model artifact root must be a real directory.'
    }
    $manifestPath = Join-Path $directory.FullName 'manifest.json'
    $manifestItem = Get-Item -LiteralPath $manifestPath -Force
    if ($manifestItem.PSIsContainer -or ($manifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Model manifest must be a regular file.'
    }
    $manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($manifestHash -cne $ExpectedManifest) { throw 'Model manifest digest mismatch.' }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema_version -cne 'lfg.encoder-onnx-artifact.v2' -or
        $manifest.files_sha256 -isnot [PSCustomObject]) {
        throw 'Unsupported model artifact manifest.'
    }
    $entries = @($manifest.files_sha256.PSObject.Properties | Sort-Object Name)
    if ($entries.Count -eq 0) { throw 'Model artifact manifest has no files.' }
    $expected = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::Ordinal)
    $windowsNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $entries) {
        $name = $entry.Name
        $digest = $entry.Value
        if ([string]::IsNullOrWhiteSpace($name) -or $name -in @('.', '..', 'manifest.json') -or
            $name.EndsWith('.') -or $name.EndsWith(' ') -or $name.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
            -not $windowsNames.Add($name) -or $digest -isnot [string] -or $digest -cnotmatch '^[0-9a-f]{64}$') {
            throw 'Invalid model artifact file entry.'
        }
        $expected.Add($name, $digest)
    }
    foreach ($required in @('tokenizer.json', 'labels.json', 'prediction-config.json', 'taxonomy.json', 'model.onnx')) {
        if (-not $expected.ContainsKey($required)) { throw "Required model artifact is missing from manifest: $required" }
    }
    $hashes = [ordered]@{}
    $hashes['manifest.json'] = $manifestHash
    foreach ($name in $expected.Keys | Sort-Object) {
        $item = Get-Item -LiteralPath (Join-Path $directory.FullName $name) -Force
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Model artifact file must be regular: $name"
        }
        $hashes[$name] = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected[$name] -cne $hashes[$name]) { throw "Model artifact digest mismatch: $name" }
    }
    return $hashes
}

Push-Location $repo
try {
    $resolved = (& git rev-parse --verify "$Commit^{commit}").Trim()
    if ($LASTEXITCODE -ne 0 -or $resolved -cne $Commit) { throw 'Commit could not be verified.' }
    if ($hasModelDirectory) {
        $modelSource = (Resolve-Path -LiteralPath $ModelArtifactDir).Path
        $modelHashes = Get-ModelHashes $modelSource $ModelManifestSha256
    }
    New-Item -ItemType Directory -Path $destination -ErrorAction Stop | Out-Null
    if ($hasModelDirectory) {
        $modelSnapshot = Join-Path $destination 'model-snapshot'
        New-Item -ItemType Directory -Path $modelSnapshot -ErrorAction Stop | Out-Null
        foreach ($name in $modelHashes.Keys) {
            Copy-Item -LiteralPath (Join-Path $modelSource $name) -Destination (Join-Path $modelSnapshot $name) -ErrorAction Stop
        }
        if ((ConvertTo-Json (Get-ModelHashes $modelSource $ModelManifestSha256) -Compress) -cne (ConvertTo-Json $modelHashes -Compress) -or
            (ConvertTo-Json (Get-ModelHashes $modelSnapshot $ModelManifestSha256) -Compress) -cne (ConvertTo-Json $modelHashes -Compress)) {
            throw 'Model artifact changed during snapshot creation.'
        }
    }
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
        $env:LFG_EMBEDDED_ONNX_DIR = if ($hasModelDirectory) { $modelSnapshot } else { $null }
        $env:LFG_EMBEDDED_ONNX_SHA256 = if ($hasModelDirectory) { $ModelManifestSha256 } else { $null }
        $features = if ($hasModelDirectory) { 'embedded-ui,embedded-onnx' } else { 'embedded-ui' }
        Push-Location $source
        try {
            Invoke-Checked { cargo build --offline --locked --release -p lfg-server --features $features --target x86_64-pc-windows-msvc } 'Cargo build failed'
        } finally { Pop-Location }
        if ($hasModelDirectory -and (ConvertTo-Json (Get-ModelHashes $modelSnapshot $ModelManifestSha256) -Compress) -cne (ConvertTo-Json $modelHashes -Compress)) {
            throw 'Model snapshot changed during the build.'
        }
        $binary = Join-Path $target 'x86_64-pc-windows-msvc\release\lfg-server.exe'
        $outputs += [ordered]@{
            binary = $binary
            sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
            bytes = (Get-Item -LiteralPath $binary).Length
            assets = (Get-AssetHashes $assets)
        }
    }
    $assetsEqual = (ConvertTo-Json $outputs[0].assets -Compress) -ceq (ConvertTo-Json $outputs[1].assets -Compress)
    if ((Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $builderSha256) {
        throw 'Build script changed during the build.'
    }
    $report = [ordered]@{
        schemaVersion = if ($hasModelDirectory) { 'lfg.windows-embedded-model-build.v1' } else { 'lfg.windows-baseline-build.v1' }
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
    if ($hasModelDirectory) {
        $report['features'] = 'embedded-ui,embedded-onnx'
        $report['builderSha256'] = $builderSha256
        $report['modelManifestSha256'] = $ModelManifestSha256
        $report['modelFilesSha256'] = $modelHashes
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
