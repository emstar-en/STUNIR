$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2

function Show-Usage {
  Write-Error "Usage: scripts/build_targets.ps1 <targets_csv> [--require-deps|--allow-missing-deps] [--lockfile <path>] [--allowlist-json <path>] [--step-env-json <path>] [--timeout-sec <n>] [--stdout <path>] [--stderr <path>] [--process-receipt <path>]"
}

if ($args.Count -lt 1) { Show-Usage; exit 2 }

# Locate repo root relative to this file
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot ".." )).Path

$buildSh = Join-Path $repoRoot "scripts/build.sh"
if (-not (Test-Path -LiteralPath $buildSh)) {
  throw "scripts/build.sh not found. Expected at: $buildSh"
}

# Load process isolation helpers
$proc = Join-Path $PSScriptRoot "stunir_process.ps1"
if (-not (Test-Path -LiteralPath $proc)) { throw "scripts/stunir_process.ps1 not found. Expected at: $proc" }
. $proc

$targets = $args[0]

$requireDeps = $null
$lockfile = "build/local_toolchain.lock.json"
$allowlistJson = "spec/env/host_env_allowlist.windows.runtime.json"
$stepEnvJson = ""
$timeoutSec = 0
$stdoutFile = ""
$stderrFile = ""
$processReceipt = ""

function Resolve-RepoRelative {
  param([string]$Path)
  if (-not $Path) { return $Path }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return (Join-Path $repoRoot $Path)
}

$i = 1
while ($i -lt $args.Count) {
  $f = $args[$i]
  switch ($f) {
    "--require-deps" { $requireDeps = $true; $i += 1; continue }
    "--allow-missing-deps" { $requireDeps = $false; $i += 1; continue }

    "--lockfile" {
      if ($i + 1 -ge $args.Count) { throw "--lockfile requires a value" }
      $lockfile = $args[$i+1]
      $i += 2
      continue
    }

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
  if ($env:STUNIR_REQUIRE_DEPS) { $requireDeps = ($env:STUNIR_REQUIRE_DEPS -eq "1") } else { $requireDeps = $false }
}

$lockfilePath = Resolve-RepoRelative -Path $lockfile
if (-not (Test-Path -LiteralPath $lockfilePath)) {
  throw "Toolchain lockfile not found: $lockfilePath (run discovery first)"
}

$allowlistPath = Resolve-RepoRelative -Path $allowlistJson
$stepEnvPath = Resolve-RepoRelative -Path $stepEnvJson
$stdoutPath = Resolve-RepoRelative -Path $stdoutFile
$stderrPath = Resolve-RepoRelative -Path $stderrFile
$receiptPath = Resolve-RepoRelative -Path $processReceipt

# Read allowlist (inherit keys)
$inheritKeys = @()
if ($allowlistPath -and $allowlistPath.Trim().Length -gt 0) {
  $allow = Read-StunirJsonFile -Path $allowlistPath
  if ($allow.inherit) { foreach ($k in $allow.inherit) { $inheritKeys += "$k" } }
}

# Read lockfile
$lockRaw = Get-Content -LiteralPath $lockfilePath -Raw
$lock = $lockRaw | ConvertFrom-Json -Depth 200

if (-not $lock.tools) { throw "Invalid lockfile: missing .tools" }
if (-not $lock.tools.bash) { throw "Invalid lockfile: missing tools.bash" }

$bashExe = $lock.tools.bash.path
if (-not $bashExe -or -not (Test-Path -LiteralPath $bashExe)) {
  throw "Locked bash path missing or not found: $bashExe"
}

# Compute substrate PATH
$substrateDirs = @()
if ($lock.shell_substrate_path_dirs) {
  foreach ($d in $lock.shell_substrate_path_dirs) { if ($d) { $substrateDirs += [string]$d } }
} elseif ($lock.path_dirs) {
  foreach ($d in $lock.path_dirs) { if ($d) { $substrateDirs += [string]$d } }
}

$substrateDirs = $substrateDirs | Select-Object -Unique
if ($substrateDirs.Count -eq 0) {
  throw "Invalid lockfile for Hybrid Strict: missing shell_substrate_path_dirs (or path_dirs)"
}

$pathValue = ($substrateDirs -join ';')

# Build-local TEMP/TMP
$tmpDir = Join-Path $repoRoot 'build/tmp'
if (-not (Test-Path -LiteralPath $tmpDir)) { New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null }

# Step env overlay
$stepEnv = @{}

if ($stepEnvPath -and $stepEnvPath.Trim().Length -gt 0) {
  if (-not (Test-Path -LiteralPath $stepEnvPath)) { throw "Step env JSON not found: $stepEnvPath" }
  $envObj = Read-StunirJsonFile -Path $stepEnvPath
  if ($envObj.env) { $envObj = $envObj.env }
  foreach ($p in $envObj.PSObject.Properties) { $stepEnv[$p.Name] = "$($p.Value)" }
}

# Policy knobs
if (-not $stepEnv.ContainsKey('STUNIR_STRICT')) { $stepEnv['STUNIR_STRICT'] = '1' }

$stepEnv['STUNIR_OUTPUT_TARGETS'] = $targets
$stepEnv['STUNIR_REQUIRE_DEPS'] = $(if ($requireDeps) { '1' } else { '0' })

# Hybrid Strict: PATH for shell substrate only
$stepEnv['PATH'] = $pathValue

# TEMP/TMP redirected
$stepEnv['TEMP'] = $tmpDir
$stepEnv['TMP'] = $tmpDir

# Inject absolute tool paths: STUNIR_TOOL_<NAME>
foreach ($toolProp in $lock.tools.PSObject.Properties) {
  $name = [string]$toolProp.Name
  $tool = $toolProp.Value
  if (-not $tool.path) { continue }

  $san = ($name.ToUpperInvariant() -replace '[^A-Z0-9]', '_')
  $envName = "STUNIR_TOOL_$san"
  $stepEnv[$envName] = [string]$tool.path
}

$envClean = New-StunirCleanEnvironment -StepEnv $stepEnv -InheritKeys $inheritKeys

$result = Invoke-StunirProcess \
  -Exe $bashExe \
  -Args @('scripts/build.sh') \
  -Env $envClean \
  -WorkingDirectory $repoRoot \
  -TimeoutSec $timeoutSec \
  -StdoutFile $(if($stdoutFile){$stdoutPath}else{$null}) \
  -StderrFile $(if($stderrFile){$stderrPath}else{$null})

if ($processReceipt -and $processReceipt.Trim().Length -gt 0) {
  Write-StunirProcessReceipt -Path $receiptPath -Exe $bashExe -Args @('scripts/build.sh') -Env $envClean -ExitCode $result.ExitCode -DurationMs $result.DurationMs
}

if ($result.ExitCode -ne 0) {
  if (-not $stderrFile) {
    if ($result.Stderr) { [Console]::Error.WriteLine($result.Stderr) }
  }
  exit $result.ExitCode
}
