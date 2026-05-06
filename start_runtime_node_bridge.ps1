param(
    [int]$IntervalSeconds = 90
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) { Join-Path $root ".venv\Scripts\python.exe" } else { "python" }
$reportsDir = Join-Path $root "outputs\runtime_node_bridge"
$pidPath = Join-Path $reportsDir "bridge.pid"
$stdoutPath = Join-Path $reportsDir "bridge.out.log"
$stderrPath = Join-Path $reportsDir "bridge.err.log"
New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null

if (Test-Path $pidPath) {
    $existingPid = Get-Content $pidPath -ErrorAction SilentlyContinue
    if ($existingPid) {
        $process = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Output "Runtime Node bridge is already running with PID $existingPid"
            exit 0
        }
    }
}

$args = @(
    "-u",
    ".\runtime_node_bridge.py",
    "--loop",
    "--interval-seconds", $IntervalSeconds
)

$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -Path $pidPath -Encoding ascii
Write-Output "Started Runtime Node bridge with PID $($process.Id)"
