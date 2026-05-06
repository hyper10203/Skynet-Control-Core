param(
    [int]$Port = 8501,
    [string]$BindAddress = "0.0.0.0",
    [switch]$NoBrowser,
    [switch]$SkipBridge
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) { Join-Path $root ".venv\Scripts\python.exe" } else { "python" }

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$DefaultValue
    )
    if (-not (Test-Path $Path)) {
        return $DefaultValue
    }
    $line = Get-Content $Path -ErrorAction SilentlyContinue | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1
    if (-not $line) {
        return $DefaultValue
    }
    return ($line -split "=", 2)[1].Trim()
}

if (-not $SkipBridge) {
    $bridgeInterval = Get-EnvValue -Path (Join-Path $root ".env") -Key "AXIOMGRAPH_BRIDGE_LOOP_SECONDS" -DefaultValue "90"
    Start-Process -FilePath "powershell" -ArgumentList @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $root "start_runtime_node_bridge.ps1"),
        "-IntervalSeconds", $bridgeInterval
    ) -WorkingDirectory $root -WindowStyle Hidden
}

$args = @(
    "-m",
    "streamlit",
    "run",
    ".\dashboard.py",
    "--server.address", $BindAddress,
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
