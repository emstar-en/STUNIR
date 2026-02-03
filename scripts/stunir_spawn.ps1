# STUNIR: CLI wrapper to run a command in a clean environment (PowerShell 5.1+ compatible)
# The launched process receives ONLY (allowlisted inherited vars) + (explicit step vars).

Set-StrictMode -Version 2

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'stunir_process.ps1')

param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [string[]]$Args = @(),
    [string]$AllowlistJson = 'spec/env/host_env_allowlist.windows.json',
    [string]$StepEnvJson = '',
    [string]$WorkingDirectory = '',
    [string]$StdoutFile = '',
    [string]$StderrFile = '',
    [string]$ReceiptJson = '',
    [int]$TimeoutSec = 0
)

$inheritKeys = @()
if ($AllowlistJson -and $AllowlistJson.Trim().Length -gt 0) {
    $allow = Read-StunirJsonFile -Path $AllowlistJson
    if ($allow.inherit) {
        foreach ($k in $allow.inherit) { $inheritKeys += "$k" }
    }
}

$stepEnv = @{}
if ($StepEnvJson -and $StepEnvJson.Trim().Length -gt 0) {
    $envObj = Read-StunirJsonFile -Path $StepEnvJson
    if ($envObj.env) { $envObj = $envObj.env }
    foreach ($p in $envObj.PSObject.Properties) {
        $stepEnv[$p.Name] = "$($p.Value)"
    }
}

$envClean = New-StunirCleanEnvironment -StepEnv $stepEnv -InheritKeys $inheritKeys

$wd = $null
if ($WorkingDirectory -and $WorkingDirectory.Trim().Length -gt 0) { $wd = $WorkingDirectory }

$stdout = $null
if ($StdoutFile -and $StdoutFile.Trim().Length -gt 0) { $stdout = $StdoutFile }

$stderr = $null
if ($StderrFile -and $StderrFile.Trim().Length -gt 0) { $stderr = $StderrFile }

$result = Invoke-StunirProcess -Exe $Exe -Args $Args -Env $envClean -WorkingDirectory $wd -TimeoutSec $TimeoutSec -StdoutFile $stdout -StderrFile $stderr

if ($ReceiptJson -and $ReceiptJson.Trim().Length -gt 0) {
    Write-StunirProcessReceipt -Path $ReceiptJson -Exe $Exe -Args $Args -Env $envClean -ExitCode $result.ExitCode -DurationMs $result.DurationMs
}

if ($result.ExitCode -ne 0) {
    if (-not $StderrFile) {
        if ($result.Stderr) { [Console]::Error.WriteLine($result.Stderr) }
    }
    exit $result.ExitCode
}
