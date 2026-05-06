param(
    [switch]$AllowSubmit,
    [int]$History = 10,
    [double]$MinLocalDelta = 0.0,
    [int]$SleepSeconds = 60,
    [int]$MaxPendingSubmissions = 2
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path (Join-Path $root ".venv\Scripts\python.exe")) { Join-Path $root ".venv\Scripts\python.exe" } else { "python" }
$reportsDir = Join-Path $root "outputs\neurogolf"
$pidPath = Join-Path $reportsDir "autonomous.pid"
$stdoutPath = Join-Path $reportsDir "daemon.out.log"
$stderrPath = Join-Path $reportsDir "daemon.err.log"
New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null

& $python (Join-Path $root "healthcheck.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Healthcheck failed. Refusing to start autonomous NeuroGolf with an unhealthy model stack."
    exit $LASTEXITCODE
}

if (Test-Path $pidPath) {
    $existingPid = Get-Content $pidPath -ErrorAction SilentlyContinue
    if ($existingPid) {
        $process = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Output "Autonomous NeuroGolf is already running with PID $existingPid"
            exit 0
        }
    }
}

$args = @(
    "-u",
    ".\autonomous_neurogolf.py",
    "--loop",
    "--history", $History,
    "--min-local-delta", $MinLocalDelta,
    "--sleep-seconds", $SleepSeconds,
    "--max-pending-submissions", $MaxPendingSubmissions
)
if ($AllowSubmit) {
    $args += "--allow-submit"
}

$process = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -Path $pidPath -Encoding ascii
Write-Output "Started autonomous NeuroGolf daemon with PID $($process.Id)"
