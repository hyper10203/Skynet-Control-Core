param(
    [switch]$AllowSubmit,
    [int]$History = 10,
    [double]$MinLocalDelta = 40.0,
    [int]$SleepSeconds = 1800,
    [int]$MaxPendingSubmissions = 1
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

powershell -ExecutionPolicy Bypass -File (Join-Path $root "stop_autonomous_neurogolf.ps1") | Out-Null
Start-Sleep -Milliseconds 300

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "start_autonomous_neurogolf.ps1"),
    "-History", $History,
    "-MinLocalDelta", $MinLocalDelta,
    "-SleepSeconds", $SleepSeconds,
    "-MaxPendingSubmissions", $MaxPendingSubmissions
)

if ($AllowSubmit) {
    $args += "-AllowSubmit"
}

powershell @args | Out-Null
