param([Parameter(Mandatory = $true)][string]$Config)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$OutputEncoding = New-Object System.Text.UTF8Encoding $false
$settings = Get-Content -Raw -LiteralPath $Config | ConvertFrom-Json
if ($settings.schema_version -ne 'lfg.isolated-runner-supervisor.v1' -or
    $settings.repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$' -or
    $settings.runner_label -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or
    $settings.runner_label -in @('self-hosted', 'linux', 'windows', 'macos', 'x64', 'arm64') -or
    $settings.image -notmatch '^[A-Za-z0-9][A-Za-z0-9._:/@-]*$' -or
    $settings.runner_group_id -lt 1 -or $settings.cpus -lt 1 -or $settings.memory_mib -lt 1 -or
    $settings.pids_limit -lt 1 -or $settings.cargo_jobs -lt 1 -or $settings.retry_seconds -lt 1) {
    throw 'Invalid isolated runner supervisor configuration.'
}

# A pull request from a fork can edit a workflow to request this runner's label,
# so register runners only for a repository that is private right now.
function Assert-PrivateRepository {
    $private = & gh api ("repos/$($settings.repository)") --jq '.private'
    if ($LASTEXITCODE -ne 0) { throw 'Could not read the repository visibility.' }
    if ($private -ne 'true') {
        throw "Refusing to register a self-hosted runner: $($settings.repository) is not private, so pull requests from forks could run code on this host."
    }
}

Assert-PrivateRepository
while ($true) {
    Assert-PrivateRepository
    $runnerName = 'lfg-jit-' + [guid]::NewGuid().ToString('N')
    $runnerId = $null
    $request = @{name = $runnerName; runner_group_id = [int]$settings.runner_group_id;
                 labels = @([string]$settings.runner_label); work_folder = '_work'} | ConvertTo-Json -Compress
    try {
        $response = $request | & gh api --method POST ("repos/$($settings.repository)/actions/runners/generate-jitconfig") --input -
        if ($LASTEXITCODE -ne 0) { throw 'GitHub runner registration failed.' }
        $registration = $response | ConvertFrom-Json
        $runnerId = [int64]$registration.runner.id
        $jit = $registration.encoded_jit_config
        if (-not $jit) { throw 'GitHub returned no runner configuration.' }
        Write-Output "Starting disposable runner $runnerName."
        $jit | & docker run --rm -i --init --name $runnerName `
            --cpus ([string]$settings.cpus) --memory ("$($settings.memory_mib)m") `
            --pids-limit ([string]$settings.pids_limit) --security-opt no-new-privileges `
            --cap-drop ALL --env ("CARGO_BUILD_JOBS=$($settings.cargo_jobs)") ([string]$settings.image)
        if ($LASTEXITCODE -ne 0) { Write-Output "Runner $runnerName exited with code $LASTEXITCODE." }
    } catch {
        Write-Output 'Runner setup failed; check GitHub access and Docker availability.'
    } finally {
        if ($runnerId) {
            try {
                $running = & docker ps -q --filter ("name=^/$runnerName$")
                if (-not $running) {
                    & gh api --method DELETE ("repos/$($settings.repository)/actions/runners/$runnerId") 2>$null | Out-Null
                }
            } catch { Write-Output "Could not confirm cleanup for runner $runnerName." }
        }
        $jit = $null
        $response = $null
        $registration = $null
    }
    Start-Sleep -Seconds ([int]$settings.retry_seconds)
}
