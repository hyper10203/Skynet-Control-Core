param(
    [int]$Limit = 5,
    [int]$CritiqueRounds = 2,
    [string]$Python = "python"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $Python = ".\.venv\Scripts\python.exe"
    }

    & $Python .\healthcheck.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $Python .\main.py --limit $Limit --critique-rounds $CritiqueRounds
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
