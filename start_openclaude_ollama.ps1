param(
    [string]$Model = "",
    [string]$Provider = "",
    [string]$Prompt = ""
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$DefaultValue
    )
    if (-not (Test-Path $Path)) {
        return $DefaultValue
    }
    $line = Get-Content $Path -ErrorAction SilentlyContinue | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1
    if (-not $line) {
        return $DefaultValue
    }
    return ($line -split "=", 2)[1].Trim()
}

$envPath = Join-Path $root ".env"
$openclaudeBin = Get-EnvValue -Path $envPath -Key "OPENCLAUDE_BIN" -DefaultValue "openclaude.cmd"
if (-not $Provider) {
    $Provider = Get-EnvValue -Path $envPath -Key "OPENCLAUDE_PROVIDER" -DefaultValue "ollama"
}
if (-not $Model) {
    $Model = Get-EnvValue -Path $envPath -Key "OPENCLAUDE_DEFAULT_MODEL" -DefaultValue "qwen2.5-coder:latest"
}

$args = @(
    "--provider", $Provider,
    "--model", $Model
)
if ($Prompt) {
    $args += @("-p", $Prompt)
}

Start-Process -FilePath $openclaudeBin -ArgumentList $args -WorkingDirectory $root
Write-Output "Started OpenClaude with provider $Provider and model $Model"
