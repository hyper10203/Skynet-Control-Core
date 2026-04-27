param(
    [switch]$AllowSubmit,
    [int]$History = 10,
    [double]$MinLocalDelta = 40.0,
    [switch]$Loop,
    [int]$SleepSeconds = 1800,
    [int]$MaxPendingSubmissions = 1
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    $python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
    & $python .\healthcheck.py
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $args = @("-u", ".\autonomous_neurogolf.py", "--history", $History, "--min-local-delta", $MinLocalDelta, "--max-pending-submissions", $MaxPendingSubmissions)
    if ($AllowSubmit) {
        $args += "--allow-submit"
    }
    if ($Loop) {
        $args += @("--loop", "--sleep-seconds", $SleepSeconds)
    }
    & $python @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
