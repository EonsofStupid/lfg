param(
    [ValidateSet(
        "BuildBinary",
        "BuildUi",
        "BuildImage",
        "GenerateSbom",
        "RunContainer",
        "StopContainer",
        "SmokeTest",
        "MlInstall",
        "MlValidate",
        "DistillTemplate",
        "DistillValidate",
        "DistillBuild",
        "All"
    )]
    [string]$Action = "All",
    [string]$ImageName = "lfg:latest",
    [string]$ContainerName = "lfg-server",
    [int]$HostPort = 4317,
    [string]$BindAddress = "127.0.0.1:4317",
    [string]$PythonCommand = "",
    [string]$BuildVersion = "0.0.0-dev",
    [string]$BuildRevision = "",
    [string]$SbomOutputPath = "output\\sbom\\lfg.spdx",
    [string]$DistillPlan = "",
    [string]$DistillOutputDirectory = ""
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

function Resolve-PythonCommand {
    param([string]$Candidate)
    if ($Candidate) {
        return $Candidate
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return "python"
    }
    $python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3) {
        return "python3"
    }
    throw "Python was not found on PATH. Install Python 3.10+."
}

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
}

function Build-Binary {
    Invoke-Step "Building release binary" {
        Invoke-NativeChecked -Command { cargo build --release -p lfg-server } -FailureMessage "cargo build failed"
    }
}

function Build-Ui {
    Invoke-Step "Building UI bundle" {
        Push-Location "ui/master-clyffy-web"
        try {
            Invoke-NativeChecked -Command { npm ci } -FailureMessage "npm ci failed"
            Invoke-NativeChecked -Command { npm run build } -FailureMessage "npm run build failed"
        } finally {
            Pop-Location
        }
    }
}

function Build-Image {
    $revision = $BuildRevision
    if (-not $revision) {
        $revision = (git rev-parse --verify HEAD 2>$null)
        if ($LASTEXITCODE -ne 0 -or -not $revision) {
            $revision = "unknown"
        }
    }
    $created = [DateTimeOffset]::UtcNow.ToString("o")
    Invoke-Step "Building deployment image $ImageName" {
        Invoke-NativeChecked -Command {
            docker build `
                --build-arg "BUILD_VERSION=$BuildVersion" `
                --build-arg "BUILD_REVISION=$revision" `
                --build-arg "BUILD_CREATED=$created" `
                -t $ImageName .
        } -FailureMessage "docker build failed"
    }
    Invoke-Step "Inspecting image metadata labels" {
        Invoke-NativeChecked -Command {
            docker inspect $ImageName --format "{{json .Config.Labels}}" 
        } -FailureMessage "docker inspect failed"
    }
}

function Generate-Sbom {
    Invoke-Step "Generating SBOM for $ImageName" {
        Invoke-NativeChecked -Command {
            .\scripts\generate-sbom.ps1 -ImageName $ImageName -OutputPath $SbomOutputPath
        } -FailureMessage "SBOM generation failed"
    }
}

function Stop-Container {
    $id = docker ps -a --filter "name=^${ContainerName}$" --format "{{.ID}}"
    if ($LASTEXITCODE -ne 0) {
        throw "docker ps failed while checking container $ContainerName"
    }
    if (-not $id) {
        Write-Host "Container $ContainerName is not present."
        return
    }
    Invoke-Step "Stopping/removing container $ContainerName" {
        Invoke-NativeChecked -Command { docker rm -f $ContainerName | Out-Null } -FailureMessage "docker rm -f failed for $ContainerName"
    }
}

function Run-Container {
    Stop-Container
    Invoke-Step "Running container $ContainerName on port $HostPort" {
        Invoke-NativeChecked -Command { docker run --rm -d --name $ContainerName -p "${HostPort}:4317" $ImageName | Out-Null } -FailureMessage "docker run failed"
    }
}

function Invoke-SmokeTest {
    Invoke-Step "Checking container health endpoint" {
        $url = "http://127.0.0.1:$HostPort/health"
        $deadline = (Get-Date).AddSeconds(30)
        do {
            try {
                $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 3
                if ($response.status -eq "ok") {
                    Write-Host "Health check OK: $($response | ConvertTo-Json -Compress)" -ForegroundColor Green
                    return
                }
            } catch {
                Start-Sleep -Seconds 1
            }
        } while ((Get-Date) -lt $deadline)
        throw "Health check failed at $url"
    }
}

function Ml-Install {
    $python = Resolve-PythonCommand -Candidate $PythonCommand
    Invoke-Step "Installing ML dependencies with $python" {
        Invoke-NativeChecked -Command { & $python -m pip install -r "ml/lfg-encoder/requirements.txt" } -FailureMessage "pip install failed"
    }
}

function Ml-Validate {
    $python = Resolve-PythonCommand -Candidate $PythonCommand
    Invoke-Step "Validating classifier dataset" {
        Invoke-NativeChecked -Command { & $python "ml/lfg-encoder/train.py" validate } -FailureMessage "train.py validate failed"
    }
}

function Distill-Template {
    $python = Resolve-PythonCommand -Candidate $PythonCommand
    Invoke-Step "Creating teacher trace template" {
        Invoke-NativeChecked -Command { & $python "ml/lfg-encoder/distill.py" init-template } -FailureMessage "distill init-template failed"
    }
}

function Distill-Validate {
    $python = Resolve-PythonCommand -Candidate $PythonCommand
    $traces = "ml/lfg-encoder/dataset/teacher-traces.v1.jsonl"
    if (-not (Test-Path $traces)) {
        throw "$traces is missing. Run DistillTemplate first."
    }
    Invoke-Step "Validating teacher traces" {
        Invoke-NativeChecked -Command { & $python "ml/lfg-encoder/distill.py" validate-traces --traces $traces } -FailureMessage "distill validate-traces failed"
    }
}

function Distill-Build {
    $python = Resolve-PythonCommand -Candidate $PythonCommand
    if (-not $DistillPlan -or -not $DistillOutputDirectory) {
        throw "DistillBuild requires -DistillPlan and -DistillOutputDirectory."
    }
    if (-not (Test-Path -LiteralPath $DistillPlan -PathType Leaf)) {
        throw "The selected distillation plan is missing."
    }
    Invoke-Step "Preparing private SFT dataset" {
        Invoke-NativeChecked -Command {
            & $python "ml/lfg-encoder/distill.py" build-sft --plan $DistillPlan --output-dir $DistillOutputDirectory
        } -FailureMessage "distill build-sft failed"
    }
}

switch ($Action) {
    "BuildBinary" { Build-Binary }
    "BuildUi" { Build-Ui }
    "BuildImage" { Build-Image }
    "GenerateSbom" { Generate-Sbom }
    "RunContainer" { Run-Container }
    "StopContainer" { Stop-Container }
    "SmokeTest" { Invoke-SmokeTest }
    "MlInstall" { Ml-Install }
    "MlValidate" { Ml-Validate }
    "DistillTemplate" { Distill-Template }
    "DistillValidate" { Distill-Validate }
    "DistillBuild" { Distill-Build }
    "All" {
        Build-Binary
        Build-Ui
        Build-Image
        Generate-Sbom
        Run-Container
        Invoke-SmokeTest
    }
}
