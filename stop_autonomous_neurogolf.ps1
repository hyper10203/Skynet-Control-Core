param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $root "outputs\neurogolf\autonomous.pid"
if (-not (Test-Path $pidPath)) {
    Write-Output "No PID file found."
    exit 0
}

$daemonPid = Get-Content $pidPath -ErrorAction SilentlyContinue
if (-not $daemonPid) {
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
    Write-Output "PID file was empty."
    exit 0
}

$process = Get-Process -Id $daemonPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $daemonPid -Force
    Write-Output "Stopped autonomous NeuroGolf daemon PID $daemonPid"
} else {
    Write-Output "Process $daemonPid was not running."
}

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*autonomous_neurogolf.py*" } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
