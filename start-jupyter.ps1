$venvPath = Join-Path $PSScriptRoot ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment not found at $venvPath"
    exit 1
}

& $activateScript
uv run --with jupyter jupyter lab
