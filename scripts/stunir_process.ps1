# STUNIR: process-isolated invocation helpers (PowerShell 5.1+ compatible)
# Design: spawn new processes; construct a clean environment explicitly.

Set-StrictMode -Version 2

function Resolve-StunirPowerShellExe {
    param(
        [switch]$PreferCore
    )

    if ($env:STUNIR_POWERSHELL_EXE -and (Test-Path -LiteralPath $env:STUNIR_POWERSHELL_EXE)) {
        return (Resolve-Path -LiteralPath $env:STUNIR_POWERSHELL_EXE).Path
    }

    try {
        $candidate = Join-Path -Path $PSHOME -ChildPath 'pwsh.exe'
        if ($PreferCore -and (Test-Path -LiteralPath $candidate)) { return (Resolve-Path -LiteralPath $candidate).Path }

        $candidate = Join-Path -Path $PSHOME -ChildPath 'powershell.exe'
        if (-not $PreferCore -and (Test-Path -LiteralPath $candidate)) { return (Resolve-Path -LiteralPath $candidate).Path }

        $candidate = Join-Path -Path $PSHOME -ChildPath 'pwsh.exe'
        if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }

        $candidate = Join-Path -Path $PSHOME -ChildPath 'powershell.exe'
        if (Test-Path -LiteralPath $candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    } catch {
        # fall through
    }

    $cmd = Get-Command -Name 'pwsh' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $cmd = Get-Command -Name 'powershell' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw "Unable to locate a PowerShell executable (pwsh or powershell)."
}

function Resolve-StunirCommand {
    param([Parameter(Mandatory=$true)][string]$Name)

    $cmd = Get-Command -Name $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Command not found: $Name"
    }

    if ($cmd.Source) { return $cmd.Source }
    if ($cmd.Path) { return $cmd.Path }

    throw "Unable to resolve command path for: $Name"
}

function Read-StunirJsonFile {
    param([Parameter(Mandatory=$true)][string]$Path)

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
    param([Parameter(Mandatory=$true)][byte[]]$Bytes)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $sha.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-StunirUtf8Bytes {
    param([Parameter(Mandatory=$true)][string]$Text)
    return [System.Text.Encoding]::UTF8.GetBytes($Text)
}

function New-StunirCleanEnvironment {
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
        if ($null -eq $v) { continue }
        $envOut["$k"] = "$v"
    }

    return $envOut
}

function Format-StunirArgument {
    param([Parameter(Mandatory=$true)][string]$Arg)

    if ($Arg -notmatch '[\s\"]') { return $Arg }
    $escaped = $Arg -replace '"', '\\"'
    return '"' + $escaped + '"'
}

function Join-StunirArguments {
    param([string[]]$Args)

    if ($null -eq $Args) { return '' }

    $parts = @()
    foreach ($a in $Args) {
        $parts += (Format-StunirArgument -Arg ("$a"))
    }

    return ($parts -join ' ')
}

function Invoke-StunirProcess {
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

    if ($WorkingDirectory) { $psi.WorkingDirectory = $WorkingDirectory }

    try {
        $psi.EnvironmentVariables.Clear()
    } catch {
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
