# STUNIR: process-isolated invocation helpers (PowerShell 5.1+ compatible)
# Design: spawn new processes; construct a clean environment explicitly; do not dot-source child steps.
# This file is intended to be imported (dot-sourced) by entrypoints like build.ps1/build_targets.ps1.

Set-StrictMode -Version 2

function Resolve-StunirPowerShellExe {
    param(
        [switch]$PreferCore
    )

    # 1) Explicit override (lets the Spec/harness pin the host)
    if ($env:STUNIR_POWERSHELL_EXE -and (Test-Path -LiteralPath $env:STUNIR_POWERSHELL_EXE)) {
        return (Resolve-Path -LiteralPath $env:STUNIR_POWERSHELL_EXE).Path
    }

    # 2) Prefer the current host (best for compatibility)
    try {
        $candidate = Join-Path -Path $PSHOME -ChildPath 'pwsh.exe'
        if ($PreferCore -and (Test-Path -LiteralPath $candidate)) { return (Resolve-Path -LiteralPath $candidate).Path }

        $candidate = Join-Path -Path $PSHOME -ChildPath 'powershell.exe'
        if (-not $PreferCore -and (Test-Path -LiteralPath $candidate)) { return (Resolve-Path -LiteralPath $candidate).Path }

        # If the preferred one doesn't exist, still return whichever exists in PSHOME.
        $candidate = Join-Path -Path $PSHOME -ChildPath 'pwsh.exe'
        if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
        $candidate = Join-Path -Path $PSHOME -ChildPath 'powershell.exe'
        if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    } catch {
        # Fall through
    }

    # 3) Fallback to PATH-based resolution (last resort; still better than hard-failing)
    $cmd = Get-Command -Name 'pwsh' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command -Name 'powershell' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw "Unable to locate a PowerShell executable (pwsh or powershell)."
}

function Read-StunirJsonFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "JSON file not found: $Path"
    }
    $raw = Get-Content -LiteralPath $Path -Raw
    if ($null -eq $raw -or $raw.Trim().Length -eq 0) {
        throw "JSON file is empty: $Path"
    }
    return ($raw | ConvertFrom-Json -Depth 100)
}

function Get-StunirSha256Hex {
    param(
        [Parameter(Mandatory=$true)][byte[]]$Bytes
    )
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $sha.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-StunirUtf8Bytes {
    param(
        [Parameter(Mandatory=$true)][string]$Text
    )
    return [System.Text.Encoding]::UTF8.GetBytes($Text)
}

function New-StunirCleanEnvironment {
    <#
    Constructs a clean environment map.

    - Does NOT blindly inherit the parent environment.
    - Optionally inherits a small, explicit allowlist of host variables (defined in a spec JSON).
    - Overlays explicit step variables (the harness' master state for that node).

    Parameters:
      -StepEnv: hashtable of explicit key/value pairs for this step.
      -InheritKeys: list of variable names to copy from the current host environment.

    Returns: [hashtable]
    #>

    param(
        [hashtable]$StepEnv = @{},
        [string[]]$InheritKeys = @()
    )

    $envOut = @{}

    foreach ($k in $InheritKeys) {
        if ($null -ne $k -and $k.Trim().Length -gt 0) {
            $v = [System.Environment]::GetEnvironmentVariable($k)
            if ($null -ne $v) {
                $envOut[$k] = $v
            }
        }
    }

    foreach ($k in $StepEnv.Keys) {
        $v = $StepEnv[$k]
        if ($null -eq $v) {
            # Explicitly omit nulls (forces consumers to be explicit)
            continue
        }
        $envOut["$k"] = "$v"
    }

    return $envOut
}

function Format-StunirArgument {
    param([Parameter(Mandatory=$true)][string]$Arg)
    # Windows CreateProcess quoting rules (good-enough, conservative):
    # - If no whitespace or quotes, pass through.
    # - Else wrap in quotes and escape embedded quotes.
    if ($Arg -notmatch '[\s\"]') { return $Arg }
    $escaped = $Arg -replace '"', '\\"'
    return '"' + $escaped + '"'
}

function Join-StunirArguments {
    param([string[]]$Args)
    if ($null -eq $Args) { return '' }
    $parts = @()
    foreach ($a in $Args) { $parts += (Format-StunirArgument -Arg ("$a")) }
    return ($parts -join ' ')
}

function Invoke-StunirProcess {
    <#
    Spawn a process with an explicitly constructed environment.

    Returns an object:
      - ExitCode
      - Stdout
      - Stderr
      - DurationMs
      - Exe
      - Args

    Notes:
      - Uses System.Diagnostics.ProcessStartInfo for PowerShell 5.1 compatibility.
      - Redirects stdout/stderr so the harness can hash / receipt them.
    #>

    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [string[]]$Args = @(),
        [hashtable]$Env = @{},
        [string]$WorkingDirectory = $null,
        [int]$TimeoutSec = 0,
        [string]$StdoutFile = $null,
        [string]$StderrFile = $null,
        [switch]$ThrowOnNonZero
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = (Join-StunirArguments -Args $Args)
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    if ($WorkingDirectory) {
        $psi.WorkingDirectory = $WorkingDirectory
    }

    # Clear inherited env and set only what the harness provides.
    try {
        $psi.EnvironmentVariables.Clear()
    } catch {
        # Some environments may not support Clear(); remove keys manually.
        $keys = @()
        foreach ($k in $psi.EnvironmentVariables.Keys) { $keys += $k }
        foreach ($k in $keys) { $psi.EnvironmentVariables.Remove($k) | Out-Null }
    }

    foreach ($k in $Env.Keys) {
        $psi.EnvironmentVariables["$k"] = "$($Env[$k])"
    }

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $null = $p.Start()

    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()

    if ($TimeoutSec -gt 0) {
        $ok = $p.WaitForExit($TimeoutSec * 1000)
        if (-not $ok) {
            try { $p.Kill() } catch {}
            throw "Process timed out after ${TimeoutSec}s: $Exe"
        }
    } else {
        $p.WaitForExit()
    }

    $sw.Stop()

    if ($StdoutFile) { [System.IO.File]::WriteAllText($StdoutFile, $stdout, [System.Text.Encoding]::UTF8) }
    if ($StderrFile) { [System.IO.File]::WriteAllText($StderrFile, $stderr, [System.Text.Encoding]::UTF8) }

    $result = [PSCustomObject]@{
        ExitCode   = $p.ExitCode
        Stdout     = $stdout
        Stderr     = $stderr
        DurationMs = [int]$sw.ElapsedMilliseconds
        Exe        = $Exe
        Args       = $Args
    }

    if ($ThrowOnNonZero -and $result.ExitCode -ne 0) {
        $msg = "Process failed with exit code $($result.ExitCode): $Exe"
        if ($stderr -and $stderr.Trim().Length -gt 0) { $msg = $msg + "`n" + $stderr }
        throw $msg
    }

    return $result
}

function Write-StunirProcessReceipt {
    <#
    Writes a minimal receipt JSON for a process invocation.

    IMPORTANT: This receipt records env keys and SHA-256 of env values (UTF-8), not raw values.
    #>

    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Args,
        [Parameter(Mandatory=$true)][hashtable]$Env,
        [Parameter(Mandatory=$true)][int]$ExitCode,
        [Parameter(Mandatory=$true)][int]$DurationMs
    )

    $envDigests = @()
    foreach ($k in ($Env.Keys | Sort-Object)) {
        $v = "$($Env[$k])"
        $envDigests += [PSCustomObject]@{
            key = "$k"
            value_sha256 = (Get-StunirSha256Hex -Bytes (Get-StunirUtf8Bytes -Text $v))
        }
    }

    $obj = [PSCustomObject]@{
        schema = 'stunir.process_receipt.v1'
        exe = $Exe
        argv = $Args
        env = $envDigests
        exit_code = $ExitCode
        duration_ms = $DurationMs
    }

    $json = $obj | ConvertTo-Json -Depth 100
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.Encoding]::UTF8)
}
