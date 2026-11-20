param(
    [string]$Python = "python",
    [switch]$CpuOnly
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Не удалось создать окружение Python 3.11–3.13." }
    }
    $ProjectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    & $ProjectPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "Не удалось подготовить pip в окружении." }
    & $ProjectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Не удалось обновить pip." }
    $TorchIndex = if ($CpuOnly) { "https://download.pytorch.org/whl/cpu" } else { "https://download.pytorch.org/whl/cu128" }
    & $ProjectPython -m pip install "torch==2.9.1" --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить PyTorch." }
    & $ProjectPython -m pip install -e ".[model,dev]"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось установить проект." }
    Write-Host "Окружение готово. Для автоматического распознавания скачайте модель командой .\.venv\Scripts\python.exe scripts/download_model.py"
} finally {
    Pop-Location
}
