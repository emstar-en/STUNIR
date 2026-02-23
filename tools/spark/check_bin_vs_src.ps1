#!/usr/bin/env pwsh
# Verify expected binaries (from stunir_tools.gpr Main list) match bin outputs

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

$GprPath = Join-Path $ScriptDir "stunir_tools.gpr"
$BinDir = Join-Path $ScriptDir "bin"
$ReportDir = Join-Path $RepoRoot "work_artifacts\analysis\self_refine\reports"

if (-not (Test-Path $ReportDir)) {
    New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null
}

if (-not (Test-Path $GprPath)) {
    Write-Host "ERROR: Missing stunir_tools.gpr at $GprPath" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $BinDir)) {
    Write-Host "ERROR: Missing bin directory at $BinDir" -ForegroundColor Red
    exit 1
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

# Actual bins (strip .exe)
$ActualBins = Get-ChildItem -Path $BinDir -Filter "*.exe" -File | ForEach-Object { $_.BaseName } | Sort-Object -Unique

$Missing = $ExpectedBins | Where-Object { $_ -notin $ActualBins }
$Extra = $ActualBins | Where-Object { $_ -notin $ExpectedBins }

$ReportJson = Join-Path $ReportDir "bin_vs_src_report.json"
$ReportTxt = Join-Path $ReportDir "bin_vs_src_report.txt"

$Report = [ordered]@{
    timestamp = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssZ")
    gpr_path = $GprPath
    bin_dir = $BinDir
    expected_count = $ExpectedBins.Count
    actual_count = $ActualBins.Count
    missing_count = $Missing.Count
    extra_count = $Extra.Count
    missing = $Missing
    extra = $Extra
}

$Report | ConvertTo-Json -Depth 5 | Out-File -FilePath $ReportJson -Encoding UTF8

$LinesOut = @()
$LinesOut += "STUNIR bin vs src (Main list) check"
$LinesOut += "Timestamp: $($Report.timestamp)"
$LinesOut += "GPR: $GprPath"
$LinesOut += "Bin: $BinDir"
$LinesOut += ""
$LinesOut += "Expected binaries: $($Report.expected_count)"
$LinesOut += "Actual binaries:   $($Report.actual_count)"
$LinesOut += "Missing:           $($Report.missing_count)"
$LinesOut += "Extra:             $($Report.extra_count)"
$LinesOut += ""
$LinesOut += "Missing (expected but not present):"
if ($Missing.Count -eq 0) { $LinesOut += "  - None" } else { $LinesOut += ($Missing | ForEach-Object { "  - $_" }) }
$LinesOut += ""
$LinesOut += "Extra (present but not expected):"
if ($Extra.Count -eq 0) { $LinesOut += "  - None" } else { $LinesOut += ($Extra | ForEach-Object { "  - $_" }) }

$LinesOut | Out-File -FilePath $ReportTxt -Encoding UTF8

Write-Host "Report written:" -ForegroundColor Green
Write-Host "  $ReportJson"
Write-Host "  $ReportTxt"

Write-Host "Expected count: $($ExpectedBins.Count)" -ForegroundColor Yellow
Write-Host "Actual count:   $($ActualBins.Count)" -ForegroundColor Yellow
Write-Host "Missing:        $($Missing.Count)" -ForegroundColor Yellow
Write-Host "Extra:          $($Extra.Count)" -ForegroundColor Yellow

if ($Missing.Count -gt 0) { exit 2 }
exit 0
