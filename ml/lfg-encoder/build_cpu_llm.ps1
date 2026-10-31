param(
    [string]$ModelId = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    [string]$WorkDir = (Join-Path $PSScriptRoot "artifacts\cpu-llm"),
    [string]$LlamaCppDir = (Join-Path $PSScriptRoot "..\..\..\llama.cpp"),
    [ValidateSet("q4_k_m", "q3_k_m", "q2_k", "iq2_xs")]
    [string]$QuantPreset = "q2_k"
)

$ErrorActionPreference = "Stop"

function Resolve-CommandPath {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

function Find-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($candidate in $Candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

Write-Host "== LFG CPU LLM builder ==" -ForegroundColor Cyan
Write-Host "Model: $ModelId"
Write-Host "Quant preset: $QuantPreset"
Write-Host "Work dir: $WorkDir"
Write-Host "llama.cpp dir: $LlamaCppDir"

$python = Resolve-CommandPath "python"
if ($null -eq $python) {
    throw "python not found on PATH."
}

$hf = Resolve-CommandPath "huggingface-cli"
if ($null -eq $hf) {
    throw "huggingface-cli not found on PATH. Install with: pip install huggingface_hub[cli]"
}

$convertScript = Join-Path $LlamaCppDir "convert_hf_to_gguf.py"
if (-not (Test-Path $convertScript)) {
    throw "convert_hf_to_gguf.py not found at $convertScript"
}

$quantizeExe = Find-FirstExistingPath @(
    (Join-Path $LlamaCppDir "build\bin\Release\llama-quantize.exe"),
    (Join-Path $LlamaCppDir "build\bin\llama-quantize.exe")
)
if ($null -eq $quantizeExe) {
    throw "llama-quantize.exe not found in llama.cpp build output."
}

$cliExe = Find-FirstExistingPath @(
    (Join-Path $LlamaCppDir "build\bin\Release\llama-cli.exe"),
    (Join-Path $LlamaCppDir "build\bin\llama-cli.exe")
)
if ($null -eq $cliExe) {
    throw "llama-cli.exe not found in llama.cpp build output."
}

New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null

$modelSafe = $ModelId.Replace("/", "--")
$modelDir = Join-Path $WorkDir $modelSafe
$f16Path = Join-Path $WorkDir "$modelSafe.f16.gguf"
$quantPath = Join-Path $WorkDir "$modelSafe.$QuantPreset.gguf"

Write-Host ""
Write-Host "[1/4] Downloading model snapshot from Hugging Face..." -ForegroundColor Yellow
& $hf download $ModelId --local-dir $modelDir --local-dir-use-symlinks False

Write-Host ""
Write-Host "[2/4] Converting HF model to F16 GGUF..." -ForegroundColor Yellow
& $python $convertScript $modelDir --outfile $f16Path --outtype f16

Write-Host ""
Write-Host "[3/4] Quantizing GGUF ($QuantPreset)..." -ForegroundColor Yellow
& $quantizeExe $f16Path $quantPath $QuantPreset

Write-Host ""
Write-Host "[4/4] Final artifact summary..." -ForegroundColor Yellow
$artifact = Get-Item $quantPath
$sizeMb = [math]::Round(($artifact.Length / 1MB), 2)

Write-Host "Created: $($artifact.FullName)" -ForegroundColor Green
Write-Host "Size MB: $sizeMb" -ForegroundColor Green

Write-Host ""
Write-Host "CPU run example:" -ForegroundColor Cyan
Write-Host "& `"$cliExe`" -m `"$($artifact.FullName)`" -p `"Summarize complexity routing strategy in 5 bullet points.`" -n 160 -t 8 -ngl 0"

Write-Host ""
Write-Host "Notes:" -ForegroundColor Cyan
Write-Host "- q4_k_m is usually higher quality but larger."
Write-Host "- q2_k/iq2_xs is usually closer to a 450-650MB target for 2B class models."
Write-Host "- If quality is too low at target size, use a smaller distilled model at a higher-bit quant."
