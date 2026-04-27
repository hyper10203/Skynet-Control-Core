param(
    [string]$Python = "python",
    [switch]$CreateVenv = $true
)

if ($CreateVenv) {
    if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
        & $Python -m venv .venv
    }
    $Python = ".\.venv\Scripts\python.exe"
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r .\requirements.txt
