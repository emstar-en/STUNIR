\
    # STUNIR toolchain discovery bootstrap (PowerShell 5.1+)
    # Purpose: generate build/local_toolchain.lock.json without requiring Python.
    #
    # Minimum viable tool list for Windows execution (per current plan):
    #   - python
    #   - bash
    #   - git
    # Plus POSIX utilities from the bash distribution: cp, rm, mkdir (default).

    Set-StrictMode -Version 2
    $ErrorActionPreference = 'Stop'

    param(
      [string]$Out = 'build/local_toolchain.lock.json',
      [string]$AllowlistJson = 'spec/env/host_env_allowlist.windows.discovery.json',
      [string]$PosixUtils = 'cp,rm,mkdir',
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

    function Pick-Exe([string]$logical, [string]$overrideEnv, [string[]]$fallbacks) {
      if ($overrideEnv -and $env:$overrideEnv) {
        $v = $env:$overrideEnv
        if (Test-Path -LiteralPath $v) { return (Resolve-Path -LiteralPath $v).Path }
      }
      foreach ($n in $fallbacks) {
        $cmd = Get-Command -Name $n -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source) { return $cmd.Source }
      }
      return $null
    }

    function Tool-Record([string]$logical, [string]$path) {
      return [ordered]@{
        path = (Norm-AbsPath $path)
        sha256 = (Sha256-File $path)
        version_string = (Probe-Version $path @('--version'))
        verification_strategy = 'single_binary'
      }
    }

    function Find-CoreutilsFromBash([string]$bashPath, [string[]]$utils) {
      $bashDir = Split-Path -Parent $bashPath
      $cand = @(
        $bashDir,
        (Resolve-Path -LiteralPath (Join-Path $bashDir '..\usr\bin') -ErrorAction SilentlyContinue).Path,
        (Resolve-Path -LiteralPath (Join-Path $bashDir '..\..\usr\bin') -ErrorAction SilentlyContinue).Path
      ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

      $records = [ordered]@{}
      $dirs = @()

      foreach ($u in $utils) {
        $exeName = $u + '.exe'
        $found = $null
        foreach ($d in $cand) {
          $p = Join-Path $d $exeName
          if (Test-Path -LiteralPath $p) { $found = (Resolve-Path -LiteralPath $p).Path; break }
        }
        if ($found) {
          $records[$u] = (Tool-Record $u $found)
          $dirs += (Norm-AbsPath (Split-Path -Parent $found))
        }
      }

      return @{ records = $records; dirs = $dirs }
    }

    $strictMode = $Strict -or ($env:STUNIR_STRICT -eq '1')

    $tools = [ordered]@{}
    $missing = @()

    $py = Pick-Exe 'python' 'STUNIR_PYTHON_EXE' @('python','python3')
    if (-not $py) { if ($strictMode) { throw 'Required tool missing in strict mode: python' } else { $missing += 'python' } }
    if ($py) { $tools['python'] = (Tool-Record 'python' $py) }

    $bash = Pick-Exe 'bash' 'STUNIR_BASH_EXE' @('bash')
    if (-not $bash) { if ($strictMode) { throw 'Required tool missing in strict mode: bash' } else { $missing += 'bash' } }
    if ($bash) { $tools['bash'] = (Tool-Record 'bash' $bash) }

    $git = Pick-Exe 'git' 'STUNIR_GIT_EXE' @('git')
    if (-not $git) { if ($strictMode) { throw 'Required tool missing in strict mode: git' } else { $missing += 'git' } }
    if ($git) { $tools['git'] = (Tool-Record 'git' $git) }

    $pathDirs = @()
    foreach ($k in @('python','bash','git')) {
      if ($tools.Contains($k)) {
        $dir = Split-Path -Parent ($tools[$k].path)
        $pathDirs += ($dir -replace '\\','/')
      }
    }

    $utils = @()
    foreach ($u in ($PosixUtils -split ',')) {
      $t = $u.Trim()
      if ($t) { $utils += $t }
    }

    if ($bash -and $utils.Count -gt 0) {
      $cu = Find-CoreutilsFromBash $bash $utils
      foreach ($name in $cu.records.Keys) {
        $tools[$name] = $cu.records[$name]
      }
      $pathDirs += $cu.dirs

      if ($cu.records.Count -gt 0) {
        if (-not $tools['bash'].Contains('critical_dependencies')) {
          $tools['bash']['critical_dependencies'] = @()
        }
        $tools['bash']['critical_dependencies'] = @($tools['bash']['critical_dependencies'] + $cu.records.Keys) | Sort-Object -Unique
      }

      if ($strictMode) {
        foreach ($u in $utils) {
          if (-not $tools.Contains($u)) { throw "Required POSIX utility missing in strict mode: $u" }
        }
      }
    }

    $pathDirs = $pathDirs | Where-Object { $_ } | Select-Object -Unique | Sort-Object

    # Snapshot env (hash by default; expand STUNIR_* wildcard)
    $envSnap = [ordered]@{ mode = $SnapshotEnv; variables = [ordered]@{} }
    if ($SnapshotEnv -ne 'none' -and (Test-Path -LiteralPath $AllowlistJson)) {
      $allow = (Get-Content -LiteralPath $AllowlistJson -Raw) | ConvertFrom-Json -Depth 50
      $inherit = @()
      foreach ($k in $allow.inherit) { $inherit += [string]$k }

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
        if ($SnapshotEnv -eq 'raw') {
          $envSnap.variables[$k] = $v
        } else {
          $envSnap.variables[$k] = (Sha256-Text $v)
        }
      }
    }

    $status = 'OK'
    if ($missing.Count -gt 0) { $status = (if ($strictMode) { 'FAILED' } else { 'TAINTED' }) }

    $obj = [ordered]@{
      schema = 'stunir.toolchain_lock.v1'
      platform = [ordered]@{
        os = 'nt'
        system = $env:OS
        release = ''
        machine = $env:PROCESSOR_ARCHITECTURE
      }
      path_normalization = [ordered]@{ absolute = $true; forward_slashes = $true; case_insensitive_compare = $true }
      tools = $tools
      path_dirs = $pathDirs
      environment_snapshot = $envSnap
      status = $status
      missing_tools = $missing
    }

    # Write atomically
    $outPath = $Out
    $outDir = Split-Path -Parent $outPath
    if ($outDir -and -not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

    $tmp = $outPath + '.tmp'
    ($obj | ConvertTo-Json -Depth 100) + "`n" | Set-Content -LiteralPath $tmp -Encoding UTF8
    Move-Item -Force -LiteralPath $tmp -Destination $outPath

    # Ensure build/tmp exists
    if (-not (Test-Path -LiteralPath 'build/tmp')) { New-Item -ItemType Directory -Force -Path 'build/tmp' | Out-Null }
