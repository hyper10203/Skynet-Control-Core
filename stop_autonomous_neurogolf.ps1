param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $root "outputs\neurogolf\autonomous.pid"

function Stop-ProcessTree {
    param([int]$TargetProcessId)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -TargetProcessId $child.ProcessId
    }
    Stop-Process -Id $TargetProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-OllamaLoadedModels {
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaExe) {
        return
    }
    $lines = & $ollamaExe.Source ps 2>$null
    if (-not $lines) {
        return
    }
    foreach ($line in ($lines -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("NAME")) {
            continue
        }
        $modelName = ($trimmed -split "\s+")[0]
        if (-not $modelName) {
            continue
        }
        & $ollamaExe.Source stop $modelName 1>$null 2>$null
    }
}

if (-not (Test-Path $pidPath)) {
    Stop-OllamaLoadedModels
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
    Stop-ProcessTree -TargetProcessId $daemonPid
    Write-Output "Stopped autonomous NeuroGolf daemon PID $daemonPid"
} else {
    Write-Output "Process $daemonPid was not running."
}

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*autonomous_neurogolf.py*" } |
    ForEach-Object {
        Stop-ProcessTree -TargetProcessId $_.ProcessId
    }
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*start_autonomous_neurogolf.ps1*" -or $_.CommandLine -like "*restart_autonomous_neurogolf.ps1*" } |
    ForEach-Object {
        Stop-ProcessTree -TargetProcessId $_.ProcessId
    }
Stop-OllamaLoadedModels
Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
