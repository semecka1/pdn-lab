$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot ".venv\Scripts\python.exe") -m pdn_lab predict --file (Join-Path $PSScriptRoot "examples\demo.txt")
exit $LASTEXITCODE
