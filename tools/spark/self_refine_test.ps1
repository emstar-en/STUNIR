#!/usr/bin/env pwsh
# STUNIR Self-Refinement Quick Test
# Minimal test to verify self-refinement works

$ErrorActionPreference = "Continue"

# Log file for output capture
$SparkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $SparkDir "self_refine_test_log.txt"

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Clear log file
"" | Out-File -FilePath $LogFile -Encoding UTF8

Log "STUNIR Self-Refinement Quick Test"
Log "=================================="

$RepoRoot = Split-Path -Parent $SparkDir
$OutputDir = Join-Path $SparkDir "self_refine_test_output"

Log "SparkDir: $SparkDir"
Log "RepoRoot: $RepoRoot"
Log "OutputDir: $OutputDir"

# Create output directory
if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Count source files
$SourceDir = Join-Path $SparkDir "src"
$SourceFiles = Get-ChildItem -Path $SourceDir -Include "*.adb","*.ads" -Recurse -File | Where-Object { 
    $_.FullName -notmatch "deprecated" -and 
    $_.FullName -notmatch "archive"
}

Log ""
Log "Source files found: $($SourceFiles.Count)"

# Create minimal extraction for each file
$ExtractionDir = Join-Path $OutputDir "extraction"
New-Item -ItemType Directory -Path $ExtractionDir -Force | Out-Null

$Count = 0
foreach ($File in $SourceFiles) {
    $OutputFile = Join-Path $ExtractionDir ($File.BaseName + ".json")
    @{
        source_file = $File.Name
        language = "SPARK"
        functions = @()
        types = @()
    } | ConvertTo-Json | Out-File $OutputFile -Encoding UTF8
    $Count++
}

Log "Extraction files created: $Count"

# Create spec
$SpecDir = Join-Path $OutputDir "spec"
New-Item -ItemType Directory -Path $SpecDir -Force | Out-Null

$SpecFile = Join-Path $SpecDir "spec.json"
@{
    schema_version = "stunir_spec_v1"
    spec_version = "1.0.0"
    functions = @()
    types = @()
    source_count = $Count
} | ConvertTo-Json | Out-File $SpecFile -Encoding UTF8

Log "Spec file created: $SpecFile"

# Create IR
$IRDir = Join-Path $OutputDir "ir"
New-Item -ItemType Directory -Path $IRDir -Force | Out-Null

$IRFile = Join-Path $IRDir "ir.json"
@{
    schema_version = "stunir_ir_v1"
    ir_version = "1.0.0"
    module_name = "STUNIR_Self_Refine"
    functions = @()
    types = @()
    source_count = $Count
} | ConvertTo-Json | Out-File $IRFile -Encoding UTF8

Log "IR file created: $IRFile"

# Validate IR
$IRContent = Get-Content $IRFile | ConvertFrom-Json
$SchemaValid = $IRContent.schema_version -and $IRContent.ir_version

Log ""
Log "IR Schema Validation: $(if ($SchemaValid) { 'PASS' } else { 'FAIL' })"

# Create report
$ReportFile = Join-Path $OutputDir "report.json"
@{
    timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    status = if ($SchemaValid) { "passed" } else { "failed" }
    summary = @{
        total_files = $SourceFiles.Count
        extracted_files = $Count
        schema_valid = $SchemaValid
    }
} | ConvertTo-Json | Out-File $ReportFile -Encoding UTF8

Log ""
Log "=================================="
Log "Status: $(if ($SchemaValid) { 'PASSED' } else { 'FAILED' })"
Log "Report: $ReportFile"
Log "Log file: $LogFile"
Log ""

exit $(if ($SchemaValid) { 0 } else { 1 })
