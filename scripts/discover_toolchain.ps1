# STUNIR toolchain discovery bootstrap (PowerShell 5.1+)
# Purpose: generate build/local_toolchain.lock.json without requiring Python.
# Hybrid Strict: runtime PATH includes only shell substrate dir(s).

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

param(
  [string]$Out = 'build/local_toolchain.lock.json',
  [string]$AllowlistJson = 'spec/env/host_env_allowlist.windows.discovery.json',
  [string]$PosixUtils = 'cp,rm,mkdir,sed,awk',
  [ValidateSet('raw','sha256','none')][string]$SnapshotEnv = 'sha256',
  [switch]$Strict
)

function Norm-AbsPath([string]$p) {
  $rp = (Resolve-Path -LiteralPath $p).Path
  return ($rp -replace '\\', '/')
}

function Sha256-File([string]$p) {
  return (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant()
}

function Sha256-Text([string]$s) {
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($s)
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try {
    $hash = $sha.ComputeHash($bytes)
    return ([System.BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
  } finally {
    $sha.Dispose()
  }
}

function Probe-Version([string]$exe, [string[]]$args) {
  try {
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $exe
    $pinfo.UseShellExecute = $false
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.CreateNoWindow = $true
    $pinfo.Arguments = ($args -join ' ')

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $pinfo
    $null = $p.Start()
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()

    $s = (($stdout + (if ($stdout -and $stderr) {"`n"} else {""}) + $stderr)).Trim()
    if ($s) { return $s }
    return "<empty>"
  } catch {
    return "<probe_failed: $($_.Exception.GetType().Name): $($_.Exception.Message)>"
  }
}

function Pick-Exe([string]$overrideEnv, [string[]]$fallbacks) {
  if ($overrideEnv) {
    $v = [System.Environment]::GetEnvironmentVariable($overrideEnv)
    if ($v -and (Test-Path -LiteralPath $v)) { return (Resolve-Path -LiteralPath $v).Path }
  }
  foreach ($n in $fallbacks) {
    $cmd = Get-Command -Name $n -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
  }
  return $null
}

function Tool-Record([string]$path) {
  return [ordered]@{
    path = (Norm-AbsPath $path)
    sha256 = (Sha256-File $path)
    version_string = (Probe-Version $path @('--version'))
    verification_strategy = 'single_binary'
  }
}

function Find-SubstrateDirs([string]$bashPath) {
  $bashDir = Split-Path -Parent $bashPath
  $cands = @()
  $cands += $bashDir
  $p1 = (Resolve-Path -LiteralPath (Join-Path $bashDir '..\usr\bin') -ErrorAction SilentlyContinue)
  if ($p1) { $cands += $p1.Path }
  $p2 = (Resolve-Path -LiteralPath (Join-Path $bashDir '..\..\usr\bin') -ErrorAction SilentlyContinue)
  if ($p2) { $cands += $p2.Path }
  return ($cands | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique)
}

$strictMode = $Strict -or ($env:STUNIR_STRICT -eq '1')

$tools = [ordered]@{}
$missing = @()

$py = Pick-Exe 'STUNIR_PYTHON_EXE' @('python','python3')
if (-not $py) { if ($strictMode) { throw 'Required tool missing in strict mode: python' } else { $missing += 'python' } }
if ($py) { $tools['python'] = (Tool-Record $py) }

$bash = Pick-Exe 'STUNIR_BASH_EXE' @('bash')
if (-not $bash) { if ($strictMode) { throw 'Required tool missing in strict mode: bash' } else { $missing += 'bash' } }
if ($bash) { $tools['bash'] = (Tool-Record $bash) }

$git = Pick-Exe 'STUNIR_GIT_EXE' @('git')
if (-not $git) { if ($strictMode) { throw 'Required tool missing in strict mode: git' } else { $missing += 'git' } }
if ($git) { $tools['git'] = (Tool-Record $git) }

$substrateDirs = @()
if ($bash) {
  $candDirs = Find-SubstrateDirs $bash
  $utils = @()
  foreach ($u in ($PosixUtils -split ',')) { $t = $u.Trim(); if ($t) { $utils += $t } }

  $utilDirs = @()
  $utilFound = @{}

  foreach ($u in $utils) {
    $exeName = $u + '.exe'
    $found = $null
    foreach ($d in $candDirs) {
      $p = Join-Path $d $exeName
      if (Test-Path -LiteralPath $p) { $found = (Resolve-Path -LiteralPath $p).Path; break }
    }
    if ($found) {
      $tools[$u] = (Tool-Record $found)
      $utilDirs += (Norm-AbsPath (Split-Path -Parent $found))
      $utilFound[$u] = $true
    }
  }

  if ($strictMode) {
    foreach ($u in $utils) {
      if (-not $utilFound.ContainsKey($u)) { throw "Required POSIX utility missing in strict mode: $u" }
    }
  }

  $substrateDirs = ($utilDirs | Where-Object { $_ } | Select-Object -Unique | Sort-Object)

  if ($utilFound.Count -gt 0) {
    if (-not $tools['bash'].Contains('critical_dependencies')) { $tools['bash']['critical_dependencies'] = @() }
    $tools['bash']['critical_dependencies'] = @($tools['bash']['critical_dependencies'] + $utilFound.Keys) | Sort-Object -Unique
  }
}

# Snapshot env (hash by default; expand STUNIR_* wildcard)
$envSnap = [ordered]@{ mode = $SnapshotEnv; variables = [ordered]@{} }
if ($SnapshotEnv -ne 'none' -and (Test-Path -LiteralPath $AllowlistJson)) {
  $allow = (Get-Content -LiteralPath $AllowlistJson -Raw) | ConvertFrom-Json -Depth 50
  $inherit = @(); foreach ($k in $allow.inherit) { $inherit += [string]$k }

  $expanded = @()
  foreach ($k in $inherit) {
    if ($k -eq 'STUNIR_*') {
      foreach ($ek in [System.Environment]::GetEnvironmentVariables().Keys) {
        $s = [string]$ek
        if ($s.StartsWith('STUNIR_')) { $expanded += $s }
      }
    } else {
      $expanded += $k
    }
  }

  foreach ($k in ($expanded | Sort-Object -Unique)) {
    $v = [System.Environment]::GetEnvironmentVariable($k)
    if ($null -eq $v) { continue }
    if ($SnapshotEnv -eq 'raw') { $envSnap.variables[$k] = $v } else { $envSnap.variables[$k] = (Sha256-Text $v) }
  }
}

$status = 'OK'
if ($missing.Count -gt 0) { $status = (if ($strictMode) { 'FAILED' } else { 'TAINTED' }) }

$obj = [ordered]@{
  schema = 'stunir.toolchain_lock.v1'
  platform = [ordered]@{ os = 'nt'; system = $env:OS; release = ''; machine = $env:PROCESSOR_ARCHITECTURE }
  path_normalization = [ordered]@{ absolute = $true; forward_slashes = $true; case_insensitive_compare = $true }
  tools = $tools
  shell_substrate_path_dirs = $substrateDirs
  path_dirs = $substrateDirs
  environment_snapshot = $envSnap
  status = $status
  missing_tools = $missing
}

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

$tmp = $Out + '.tmp'
($obj | ConvertTo-Json -Depth 100) + "`n" | Set-Content -LiteralPath $tmp -Encoding UTF8
Move-Item -Force -LiteralPath $tmp -Destination $Out

if (-not (Test-Path -LiteralPath 'build/tmp')) { New-Item -ItemType Directory -Force -Path 'build/tmp' | Out-Null }
