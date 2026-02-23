#!/usr/bin/env pwsh
# Move extra binaries to deprecated folder
# Extras = binaries not in stunir_tools.gpr Main list

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BinDir = Join-Path $ScriptDir "bin"
$DeprecatedDir = Join-Path $BinDir "deprecated"
$GprPath = Join-Path $ScriptDir "stunir_tools.gpr"

# Ensure deprecated dir exists
if (-not (Test-Path $DeprecatedDir)) {
    New-Item -ItemType Directory -Path $DeprecatedDir -Force | Out-Null
}

# Extract Main list from GPR
$Lines = Get-Content $GprPath
$InMain = $false
$MainEntries = @()
foreach ($Line in $Lines) {
    if ($Line -match "^\s*for\s+Main\s+use\s*\(") {
        $InMain = $true
        continue
    }
    if ($InMain -and $Line -match "\)\s*;") {
        $InMain = $false
        break
    }
    if ($InMain) {
        $Matches = [regex]::Matches($Line, '"([^"]+)"')
        foreach ($Match in $Matches) {
            $MainEntries += $Match.Groups[1].Value
        }
    }
}

$ExpectedBins = $MainEntries | ForEach-Object { $_ -replace '\.adb$', '' } | Sort-Object -Unique

# Get actual binaries
$ActualBins = Get-ChildItem -Path $BinDir -Filter "*.exe" -File

# Move extras to deprecated
$MovedCount = 0
foreach ($Bin in $ActualBins) {
    $BaseName = $Bin.BaseName
    if ($BaseName -notin $ExpectedBins) {
        $DestPath = Join-Path $DeprecatedDir $Bin.Name
        Move-Item -Path $Bin.FullName -Destination $DestPath -Force
        Write-Host "Moved: $($Bin.Name) -> deprecated/"
        $MovedCount++
    }
}

Write-Host ""
Write-Host "Total moved: $MovedCount" -ForegroundColor Yellow
Write-Host "Remaining in bin/: $(($ExpectedBins | Where-Object { Test-Path (Join-Path $BinDir "$_.exe") }).Count)" -ForegroundColor Green
