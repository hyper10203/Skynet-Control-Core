param(
    [int]$DashboardPort = 8501,
    [string]$BindAddress = "0.0.0.0",
    [switch]$NoBrowser,
    [switch]$StartAutonomy,
    [switch]$AllowSubmit,
    [int]$History = 10,
    [double]$MinLocalDelta = 0.0,
    [int]$SleepSeconds = 60,
    [int]$MaxPendingSubmissions = 2
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$dashboardArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $root "start_dashboard.ps1"),
    "-Port", $DashboardPort,
    "-BindAddress", $BindAddress
)
if ($NoBrowser) {
    $dashboardArgs += "-NoBrowser"
}

powershell @dashboardArgs | Out-Null

if ($StartAutonomy) {
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
}

Write-Output "Runtime Node launch requested."
