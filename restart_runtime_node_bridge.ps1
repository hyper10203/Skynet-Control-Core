param(
    [int]$IntervalSeconds = 90
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

powershell -ExecutionPolicy Bypass -File (Join-Path $root "stop_runtime_node_bridge.ps1") | Out-Null
Start-Sleep -Milliseconds 300

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "start_runtime_node_bridge.ps1"),
    "-IntervalSeconds", $IntervalSeconds
)

powershell @args | Out-Null
