#!/usr/bin/env pwsh
# self_refine_loop.ps1 - Iterative test and refinement loop for SPARK extractor
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

param(
    [string]$TestDir = "test_data",
    [int]$MaxIterations = 10,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$SparkDir = $PSScriptRoot
$ExePath = Join-Path $SparkDir "bin\spark_extract_main.exe"
$ResultsDir = Join-Path $SparkDir "test_output"

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "HH:mm:ss"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "Cyan" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-Extractor {
    param([string]$InputFile, [string]$OutputFile)
    
    $result = @{
        Success = $false
        ExitCode = -1
        OutputExists = $false
        StartedExists = $false
        OkExists = $false
        ErrorExists = $false
        FunctionCount = 0
        ErrorMessage = ""
        # Rich diagnostics
        InputSha256 = ""
        InputLineCount = 0
        InputBytes = 0
        StdoutContent = ""
        StderrContent = ""
        StartedContent = ""
        OkContent = ""
        ErrorContent = ""
        OutputBytes = 0
        JsonParseError = ""
    }
    
    # Compute input file metadata
    if (Test-Path $InputFile) {
        $fileInfo = Get-Item $InputFile
        $result.InputBytes = $fileInfo.Length
        $result.InputLineCount = (Get-Content $InputFile).Count
        try {
            $sha256 = Get-FileHash $InputFile -Algorithm SHA256
            $result.InputSha256 = $sha256.Hash
        } catch {
            $result.InputSha256 = "ERROR"
        }
    }
    
    # Ensure output directory exists
    $outputDir = Split-Path $OutputFile -Parent
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Remove old output files
    Remove-Item -Path "$OutputFile*" -Force -ErrorAction SilentlyContinue
    
    # Run extractor
    $stdoutFile = "$OutputFile.stdout.txt"
    $stderrFile = "$OutputFile.stderr.txt"
    
    $process = Start-Process -FilePath $ExePath `
        -ArgumentList "-i", "`"$InputFile`"", "-o", "`"$OutputFile`"", "--lang", "spark" `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutFile `
        -RedirectStandardError $stderrFile
    
    $result.ExitCode = $process.ExitCode
    
    # Capture stdout content
    if (Test-Path $stdoutFile) {
        $stdout = Get-Content $stdoutFile -Raw
        if ($stdout -and $stdout.Trim().Length -gt 0) {
            $result.StdoutContent = $stdout.Trim()
        }
    }
    
    # Capture stderr for error message - only if it contains actual errors
    if (Test-Path $stderrFile) {
        $stderr = Get-Content $stderrFile -Raw
        if ($stderr -and $stderr.Trim().Length -gt 0) {
            $result.StderrContent = $stderr.Trim()
            # Only treat as error if it contains error indicators
            $stderrTrim = $stderr.Trim()
            if ($stderrTrim -match "Error:|Exception|FAILED|failed") {
                $result.ErrorMessage = $stderrTrim
            }
        }
    }
    
    # Check output files and capture sidecar contents
    $result.OutputExists = Test-Path $OutputFile
    $result.StartedExists = Test-Path "$OutputFile.started.txt"
    $result.OkExists = Test-Path "$OutputFile.ok.txt"
    $result.ErrorExists = Test-Path "$OutputFile.error.txt"
    
    # Capture sidecar contents
    if ($result.StartedExists) {
        $result.StartedContent = (Get-Content "$OutputFile.started.txt" -Raw).Trim()
    }
    if ($result.OkExists) {
        $result.OkContent = (Get-Content "$OutputFile.ok.txt" -Raw).Trim()
    }
    if ($result.ErrorExists) {
        $errorContent = Get-Content "$OutputFile.error.txt" -Raw
        $result.ErrorContent = $errorContent.Trim()
        if ($errorContent -and $errorContent.Trim().Length -gt 0) {
            $result.ErrorMessage = $errorContent.Trim()
        }
    }
    
    if ($result.OutputExists) {
        $outputInfo = Get-Item $OutputFile
        $result.OutputBytes = $outputInfo.Length
        try {
            $json = Get-Content $OutputFile | ConvertFrom-Json
            $result.FunctionCount = $json.functions.Count
            $result.Success = $true
        } catch {
            $result.JsonParseError = $_.ToString()
            $result.ErrorMessage = "JSON parse error: $_"
        }
    }
    
    return $result
}

function Get-RepoRoot {
    $dir = $SparkDir
    while ($dir -ne $null) {
        if (Test-Path (Join-Path $dir ".git")) {
            return $dir
        }
        if (Test-Path (Join-Path $dir "STUNIR_ARCHITECTURE.mermaid")) {
            return $dir
        }
        $parent = Split-Path $dir -Parent
        if ($parent -eq $dir) { break }
        $dir = $parent
    }
    return $SparkDir
}

# Main loop
Write-Status "Starting self-refine loop for SPARK extractor"
Write-Status "SparkDir: $SparkDir"
Write-Status "ExePath: $ExePath"

# Check if extractor exists
if (-not (Test-Path $ExePath)) {
    Write-Status "Extractor not found: $ExePath" -Level "ERROR"
    Write-Status "Run: gprbuild -P spark_extract.gpr" -Level "ERROR"
    exit 1
}

# Create results directory
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null
}

# Find test files
$TestPath = Join-Path $SparkDir $TestDir
$TestFiles = @()
$TestFiles += Get-ChildItem -Path $TestPath -Filter "*.ads" -ErrorAction SilentlyContinue
$TestFiles += Get-ChildItem -Path $TestPath -Filter "*.adb" -ErrorAction SilentlyContinue

if ($TestFiles.Count -eq 0) {
    Write-Status "No test files found in $TestPath" -Level "WARN"
    exit 0
}

Write-Status "Found $($TestFiles.Count) test files"

$iteration = 0
$totalFunctions = 0
$totalErrors = 0
$results = @()

foreach ($file in $TestFiles) {
    $iteration++
    $inputPath = $file.FullName
    $outputPath = Join-Path $ResultsDir "$($file.BaseName)_extraction.json"
    
    Write-Status "Testing: $($file.Name) ($iteration/$($TestFiles.Count))"
    
    $testResult = [PSCustomObject]@{
        FileName = $file.Name
        Success = $false
        ExitCode = -1
        OutputExists = $false
        StartedExists = $false
        OkExists = $false
        ErrorExists = $false
        FunctionCount = 0
        ErrorMessage = ""
        # Rich diagnostics
        InputSha256 = ""
        InputLineCount = 0
        InputBytes = 0
        StdoutContent = ""
        StderrContent = ""
        StartedContent = ""
        OkContent = ""
        ErrorContent = ""
        OutputBytes = 0
        JsonParseError = ""
    }
    
    # Run the extractor test
    $extractorResult = Test-Extractor -InputFile $inputPath -OutputFile $outputPath
    
    # Copy all results
    $testResult.Success = $extractorResult.Success
    $testResult.ExitCode = $extractorResult.ExitCode
    $testResult.OutputExists = $extractorResult.OutputExists
    $testResult.StartedExists = $extractorResult.StartedExists
    $testResult.OkExists = $extractorResult.OkExists
    $testResult.ErrorExists = $extractorResult.ErrorExists
    $testResult.FunctionCount = $extractorResult.FunctionCount
    $testResult.ErrorMessage = $extractorResult.ErrorMessage
    # Rich diagnostics
    $testResult.InputSha256 = $extractorResult.InputSha256
    $testResult.InputLineCount = $extractorResult.InputLineCount
    $testResult.InputBytes = $extractorResult.InputBytes
    $testResult.StdoutContent = $extractorResult.StdoutContent
    $testResult.StderrContent = $extractorResult.StderrContent
    $testResult.StartedContent = $extractorResult.StartedContent
    $testResult.OkContent = $extractorResult.OkContent
    $testResult.ErrorContent = $extractorResult.ErrorContent
    $testResult.OutputBytes = $extractorResult.OutputBytes
    $testResult.JsonParseError = $extractorResult.JsonParseError
    
    $results += $testResult
    
    $totalFunctions += $testResult.FunctionCount
    
    if ($testResult.Success) {
        Write-Status "  SUCCESS: $($testResult.FunctionCount) functions extracted" -Level "SUCCESS"
    } else {
        $totalErrors++
        Write-Status "  FAILED: $($testResult.ErrorMessage)" -Level "ERROR"
    }
    
    if ($Verbose) {
        Write-Host "    ExitCode: $($testResult.ExitCode)"
        Write-Host "    OutputExists: $($testResult.OutputExists)"
        Write-Host "    InputBytes: $($testResult.InputBytes)"
        Write-Host "    InputLineCount: $($testResult.InputLineCount)"
        Write-Host "    OutputBytes: $($testResult.OutputBytes)"
        Write-Host "    InputSha256: $($testResult.InputSha256.Substring(0,16))..."
    }
}

# Summary
Write-Host ""
Write-Status "=== SUMMARY ===" -Level "INFO"
Write-Status "Files tested: $($TestFiles.Count)"
Write-Status "Total functions extracted: $totalFunctions"
Write-Status "Errors: $totalErrors"

# Write report
$repoRoot = Get-RepoRoot
$reportPath = Join-Path $repoRoot "spark_extractor_report.json"

# Include rich diagnostics in report
$report = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    extractor_version = "spark_extract_main@2026-02-23a"
    files_tested = $TestFiles.Count
    total_functions = $totalFunctions
    errors = $totalErrors
    results = $results | Select-Object FileName, Success, ExitCode, FunctionCount, ErrorMessage, InputSha256, InputLineCount, InputBytes, OutputBytes, JsonParseError
}

# Write full report
$report | ConvertTo-Json -Depth 3 | Out-File $reportPath -Encoding utf8
Write-Status "Report written to: $reportPath"

# Write detailed per-file diagnostics to separate file
$diagnosticsPath = Join-Path $repoRoot "spark_extractor_diagnostics.json"
$diagnostics = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    results = $results | Select-Object FileName, Success, ExitCode, FunctionCount, ErrorMessage, InputSha256, InputLineCount, InputBytes, OutputBytes, StdoutContent, StderrContent, StartedContent, OkContent, ErrorContent, JsonParseError
}
$diagnostics | ConvertTo-Json -Depth 4 | Out-File $diagnosticsPath -Encoding utf8
Write-Status "Diagnostics written to: $diagnosticsPath"

# Exit code
if ($totalErrors -gt 0) {
    exit 1
}
exit 0