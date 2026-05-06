param(
    [switch]$ResetAutonomyState
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

powershell -ExecutionPolicy Bypass -File (Join-Path $root "stop_runtime_node_bridge.ps1") | Out-Null

$stopArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "stop_autonomous_neurogolf.ps1")
)

powershell @stopArgs | Out-Null

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*streamlit*dashboard.py*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

if ($ResetAutonomyState) {
    $statusPath = Join-Path $root "outputs\neurogolf\status.json"
    if (Test-Path $statusPath) {
        $resetStatus = @{
            phase = "idle"
            message = "Runtime node stopped"
            progress = 0.0
            updated_at = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
            reset = $true
        } | ConvertTo-Json -Depth 4
        Set-Content -Path $statusPath -Value $resetStatus -Encoding utf8
    }
}

Write-Output "Runtime Node stop requested."
