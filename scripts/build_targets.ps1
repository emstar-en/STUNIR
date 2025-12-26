$ErrorActionPreference = "Stop"

function Show-Usage {
  Write-Error "Usage: scripts/build_targets.ps1 <targets_csv> [--require-deps|--allow-missing-deps] [--allowlist-json <path>] [--step-env-json <path>] [--timeout-sec <n>] [--stdout <path>] [--stderr <path>] [--process-receipt <path>]"
}

# Locate repo root relative to this file location
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot ".." )).Path

$buildSh = Join-Path $repoRoot "scripts/build.sh"
if (-not (Test-Path -LiteralPath $buildSh)) {
  throw "scripts/build.sh not found. Expected at: $buildSh"
}

if ($args.Count -lt 1) {
  Show-Usage
  exit 2
}

$targets = $args[0]

$requireDeps = $null
$allowlistJson = "spec/env/host_env_allowlist.windows.json"
$stepEnvJson = ""
$timeoutSec = 0
$stdoutFile = ""
$stderrFile = ""
$processReceipt = ""

$i = 1
while ($i -lt $args.Count) {
  $f = $args[$i]
  switch ($f) {
    "--require-deps" { $requireDeps = $true; $i += 1; continue }
    "--allow-missing-deps" { $requireDeps = $false; $i += 1; continue }

    "--allowlist-json" {
      if ($i + 1 -ge $args.Count) { throw "--allowlist-json requires a value" }
      $allowlistJson = $args[$i+1]
      $i += 2
      continue
    }

    "--step-env-json" {
      if ($i + 1 -ge $args.Count) { throw "--step-env-json requires a value" }
      $stepEnvJson = $args[$i+1]
      $i += 2
      continue
    }

    "--timeout-sec" {
      if ($i + 1 -ge $args.Count) { throw "--timeout-sec requires a value" }
      $timeoutSec = [int]$args[$i+1]
      $i += 2
      continue
    }

    "--stdout" {
      if ($i + 1 -ge $args.Count) { throw "--stdout requires a value" }
      $stdoutFile = $args[$i+1]
      $i += 2
      continue
    }

    "--stderr" {
      if ($i + 1 -ge $args.Count) { throw "--stderr requires a value" }
      $stderrFile = $args[$i+1]
      $i += 2
      continue
    }

    "--process-receipt" {
      if ($i + 1 -ge $args.Count) { throw "--process-receipt requires a value" }
      $processReceipt = $args[$i+1]
      $i += 2
      continue
    }

    "-h" { Show-Usage; exit 0 }
    "--help" { Show-Usage; exit 0 }

    default { throw "Unknown argument: $f" }
  }
}

if ($null -eq $requireDeps) {
  if ($env:STUNIR_REQUIRE_DEPS) {
    $requireDeps = ($env:STUNIR_REQUIRE_DEPS -eq "1")
  } else {
    $requireDeps = $false
  }
}

# Load process isolation helpers from scripts/
$proc = Join-Path $PSScriptRoot "stunir_process.ps1"
if (-not (Test-Path -LiteralPath $proc)) {
  throw "scripts/stunir_process.ps1 not found. Expected at: $proc"
}
. $proc

function Resolve-RepoRelative {
  param([string]$Path)
  if (-not $Path) { return $Path }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return (Join-Path $repoRoot $Path)
}

$allowlistJsonPath = Resolve-RepoRelative -Path $allowlistJson
$stepEnvJsonPath = Resolve-RepoRelative -Path $stepEnvJson
$stdoutPath = Resolve-RepoRelative -Path $stdoutFile
$stderrPath = Resolve-RepoRelative -Path $stderrFile
$receiptPath = Resolve-RepoRelative -Path $processReceipt

$inheritKeys = @()
if ($allowlistJsonPath -and $allowlistJsonPath.Trim().Length -gt 0) {
  $allow = Read-StunirJsonFile -Path $allowlistJsonPath
  if ($allow.inherit) {
    foreach ($k in $allow.inherit) { $inheritKeys += "$k" }
  }
}

$stepEnv = @{}
if ($stepEnvJsonPath -and $stepEnvJsonPath.Trim().Length -gt 0) {
  $envObj = Read-StunirJsonFile -Path $stepEnvJsonPath
  if ($envObj.env) { $envObj = $envObj.env }
  foreach ($p in $envObj.PSObject.Properties) {
    $stepEnv[$p.Name] = "$($p.Value)"
  }
}

$stepEnv["STUNIR_OUTPUT_TARGETS"] = $targets
$stepEnv["STUNIR_REQUIRE_DEPS"] = $(if ($requireDeps) { "1" } else { "0" })

$envClean = New-StunirCleanEnvironment -StepEnv $stepEnv -InheritKeys $inheritKeys

# Resolve bash
$bashExe = $null
if ($env:STUNIR_BASH_EXE -and (Test-Path -LiteralPath $env:STUNIR_BASH_EXE)) {
  $bashExe = (Resolve-Path -LiteralPath $env:STUNIR_BASH_EXE).Path
} else {
  $bashExe = Resolve-StunirCommand -Name "bash"
}

$result = Invoke-StunirProcess \
  -Exe $bashExe \
  -Args @("scripts/build.sh") \
  -Env $envClean \
  -WorkingDirectory $repoRoot \
  -TimeoutSec $timeoutSec \
  -StdoutFile $(if($stdoutFile){$stdoutPath}else{$null}) \
  -StderrFile $(if($stderrFile){$stderrPath}else{$null})

if ($processReceipt -and $processReceipt.Trim().Length -gt 0) {
  Write-StunirProcessReceipt -Path $receiptPath -Exe $bashExe -Args @("scripts/build.sh") -Env $envClean -ExitCode $result.ExitCode -DurationMs $result.DurationMs
}

if ($result.ExitCode -ne 0) {
  if (-not $stderrFile) {
    if ($result.Stderr) { [Console]::Error.WriteLine($result.Stderr) }
  }
  exit $result.ExitCode
}
