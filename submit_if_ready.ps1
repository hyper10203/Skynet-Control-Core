param(
    [string]$SubmissionFile,
    [double]$EstimatedScore,
    [double]$Threshold = 250,
    [string]$Message = "",
    [switch]$Force
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    $Python = "python"
    if (Test-Path ".\.venv\Scripts\python.exe") {
        $Python = ".\.venv\Scripts\python.exe"
    }

    $args = @(".\submit_to_kaggle.py")
    if ($SubmissionFile) {
        $args += @("--file", $SubmissionFile)
    }
    if ($EstimatedScore) {
        $args += @("--estimated-score", "$EstimatedScore")
    }
    if ($Threshold) {
        $args += @("--threshold", "$Threshold")
    }
    if ($Message) {
        $args += @("--message", $Message)
    }
    if ($Force) {
        $args += "--force"
    }

    & $Python @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
