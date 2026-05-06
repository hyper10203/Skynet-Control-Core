param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidPath = Join-Path $root "outputs\runtime_node_bridge\bridge.pid"

function Stop-ProcessTree {
    param([int]$TargetProcessId)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$TargetProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -TargetProcessId $child.ProcessId
    }
    Stop-Process -Id $TargetProcessId -Force -ErrorAction SilentlyContinue
}

if (Test-Path $pidPath) {
    $bridgePid = Get-Content $pidPath -ErrorAction SilentlyContinue
    if ($bridgePid) {
        $process = Get-Process -Id $bridgePid -ErrorAction SilentlyContinue
        if ($process) {
            Stop-ProcessTree -TargetProcessId $bridgePid
            Write-Output "Stopped Runtime Node bridge PID $bridgePid"
        } else {
            Write-Output "Runtime Node bridge PID $bridgePid was not running."
        }
    } else {
        Write-Output "Bridge PID file was empty."
    }
} else {
    Write-Output "No Runtime Node bridge PID file found."
}

Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*runtime_node_bridge.py*" } |
    ForEach-Object {
        Stop-ProcessTree -TargetProcessId $_.ProcessId
    }
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*start_runtime_node_bridge.ps1*" -or $_.CommandLine -like "*restart_runtime_node_bridge.ps1*" } |
    ForEach-Object {
        Stop-ProcessTree -TargetProcessId $_.ProcessId
    }
Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
