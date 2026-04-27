param(
    [int]$Port = 8501,
    [switch]$NoBrowser
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) { Join-Path $root ".venv\Scripts\python.exe" } else { "python" }
$args = @(
    "-m",
    "streamlit",
    "run",
    ".\dashboard.py",
    "--server.port", $Port,
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
)

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like '*streamlit*dashboard.py*' -and $_.CommandLine -like "*--server.port $Port*"
}
if (-not $existing) {
    Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:$Port"
}
