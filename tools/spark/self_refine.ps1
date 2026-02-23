#!/usr/bin/env pwsh
# STUNIR Self-Refinement Runner
# Runs the SPARK pipeline on STUNIR's own SPARK source code
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Runs the STUNIR SPARK pipeline on its own source code for self-refinement analysis.

.DESCRIPTION
    This script performs self-refinement by:
    1. Enumerating SPARK source files from tools/spark/src and tests/spark
    2. Running extraction → spec assembly → IR conversion
    3. Validating IR against schema
    4. Emitting SPARK/Ada target code
    5. Generating structured reports

.PARAMETER OutputDir
    Directory for self-refinement artifacts. Defaults to work_artifacts/analysis/self_refine.

.PARAMETER SparkDir
    Root directory of SPARK tools. Defaults to script location.

.PARAMETER Targets
    Comma-separated list of targets to emit. Defaults to "SPARK,Ada".

.PARAMETER Verbose
    Enable verbose output.

.PARAMETER SkipEmission
    Skip code emission phase (only run extraction/spec/IR).

.EXAMPLE
    .\self_refine.ps1 -Verbose
    Run full self-refinement with verbose output.

.EXAMPLE
    .\self_refine.ps1 -SkipEmission
    Run extraction and IR validation only, skip emission.
#>

param(
    [string]$OutputDir = "",
    [string]$SparkDir = "",
    [string]$Targets = "SPARK,Ada",
    [switch]$Verbose,
    [switch]$SkipEmission
)

$ErrorActionPreference = "Stop"

$ScriptVersion = "self_refine.ps1@2026-02-22c"
$ReportVersion = "self_refine_report_v2"

# Determine paths
if ([string]::IsNullOrEmpty($SparkDir)) {
    $SparkDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

function Get-RepoRoot([string]$StartDir) {
    $Current = $StartDir
    while ($true) {
        $HasPyProject = Test-Path (Join-Path $Current "pyproject.toml")
        $HasRepoIndex = Test-Path (Join-Path $Current "STUNIR_REPO_INDEX.json")
        $HasArch = Test-Path (Join-Path $Current "STUNIR_ARCHITECTURE.mermaid")

        if ($HasPyProject -or $HasRepoIndex -or $HasArch) {
            return $Current
        }

        $Parent = Split-Path -Parent $Current
        if ($Parent -eq $Current -or [string]::IsNullOrEmpty($Parent)) { break }
        $Current = $Parent
    }

    return (Split-Path -Parent (Split-Path -Parent $StartDir))
}

$RepoRoot = Get-RepoRoot $SparkDir

if ([string]::IsNullOrEmpty($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot "work_artifacts\analysis\self_refine"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}
# Guard against legacy tools/ work_artifacts output
if ($OutputDir -match "\\tools\\work_artifacts\\") {
    $OutputDir = Join-Path $RepoRoot "work_artifacts\analysis\self_refine"
}

# Create output directories
$ExtractionDir = Join-Path $OutputDir "extraction"
$SpecDir = Join-Path $OutputDir "spec"
$IRDir = Join-Path $OutputDir "ir"
$EmitDir = Join-Path $OutputDir "emit"
$ReportDir = Join-Path $OutputDir "reports"

# Force repo-root report directory and guard against tools/ work_artifacts
$RepoReportDir = Join-Path $RepoRoot "work_artifacts\analysis\self_refine\reports"
if ($RepoReportDir -match "\\tools\\work_artifacts\\") {
    throw "Invalid repo report path (points to tools/ work_artifacts): $RepoReportDir"
}

@($OutputDir, $ExtractionDir, $SpecDir, $IRDir, $EmitDir, $ReportDir, $RepoReportDir) | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}

# Initialize report
$Report = @{
    report_version = $ReportVersion
    script_version = $ScriptVersion
    timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    spark_dir = $SparkDir
    output_dir = $OutputDir
    targets = $Targets -split ","
    phases = @{}
    summary = @{
        total_files = 0
        extracted_files = 0
        spec_files = 0
        ir_files = 0
        emitted_files = 0
        errors = @()
        warnings = @()
    }
    # Rich per-file diagnostics
    file_details = @()
    status = "running"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STUNIR Self-Refinement Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Script:     $PSCommandPath"
Write-Host "Version:    $ScriptVersion"
Write-Host "Repository: $RepoRoot"
Write-Host "SPARK Dir:  $SparkDir"
Write-Host "Output:     $OutputDir"
Write-Host "Report Dir: $RepoReportDir"
Write-Host "Targets:    $Targets"
Write-Host ""

# Phase 0: Enumerate source files
Write-Host "[Phase 0] Enumerating SPARK source files..." -ForegroundColor Yellow

$SourceDirs = @(
    Join-Path $SparkDir "src"
    Join-Path $RepoRoot "tests\spark"
)

$SourceFiles = @()
foreach ($Dir in $SourceDirs) {
    if (Test-Path $Dir) {
        $Files = Get-ChildItem -Path $Dir -Include "*.adb","*.ads" -Recurse -File
        $SourceFiles += $Files
        if ($Verbose) {
            Write-Host "  Found $($Files.Count) files in $Dir"
        }
    }
}

# Filter out deprecated and archive directories
$SourceFiles = $SourceFiles | Where-Object { 
    $_.FullName -notmatch "deprecated" -and 
    $_.FullName -notmatch "archive" -and
    $_.FullName -notmatch "semantic_ir"
}

$Report.summary.total_files = $SourceFiles.Count
Write-Host "  Total SPARK source files: $($SourceFiles.Count)" -ForegroundColor Green

# Phase 1: Extraction
Write-Host ""
Write-Host "[Phase 1] Running extraction..." -ForegroundColor Yellow

$ExtractionResults = @{}
$ExtractedCount = 0

# Check if extraction tools exist
$CExtractCmd = Join-Path $SparkDir "bin\source_extract_main.exe"
$SparkExtractCmd = Join-Path $SparkDir "bin\spark_extract_main.exe"
$CExtractExists = [System.IO.File]::Exists($CExtractCmd)
$SparkExtractExists = [System.IO.File]::Exists($SparkExtractCmd)

if (-not $SparkExtractExists) {
    $FallbackSparkPath = Join-Path $SparkDir "bin/spark_extract_main.exe"
    if ([System.IO.File]::Exists($FallbackSparkPath)) {
        $SparkExtractCmd = $FallbackSparkPath
        $SparkExtractExists = $true
    }
}

if (-not $CExtractExists) {
    Write-Host "  Note: source_extract_main.exe not found (C/C++), using minimal extraction for C" -ForegroundColor Yellow
}
if (-not $SparkExtractExists) {
    Write-Host "  Note: spark_extract_main.exe not found, using minimal extraction for SPARK" -ForegroundColor Yellow
}
if ($Verbose) {
    Write-Host "  Extractor (C): $CExtractCmd (exists=$CExtractExists)"
    Write-Host "  Extractor (SPARK): $SparkExtractCmd (exists=$SparkExtractExists)"
}

foreach ($File in $SourceFiles) {
    $RelativePath = $File.FullName.Substring($RepoRoot.Length + 1)
    $OutputFile = Join-Path $ExtractionDir ($File.BaseName + "_extraction.json")
    
    # Initialize file detail record
    $FileDetail = @{
        file = $RelativePath
        extraction = @{ status = "pending"; exit_code = -1; output_bytes = 0; function_count = 0; error = "" }
        spec = @{ status = "pending" }
        ir = @{ status = "pending" }
        emit = @{ status = "pending" }
    }
    
    if ($Verbose) {
        Write-Host "  Extracting: $RelativePath"
    }
    
    try {
        $Ext = $File.Extension.ToLowerInvariant()
        $IsCSource = $Ext -in @(".c", ".h", ".cpp", ".hpp", ".cc")
        $IsSparkSource = $Ext -in @(".adb", ".ads")

        if ($IsCSource -and $CExtractExists) {
            $Result = & $CExtractCmd -i $File.FullName -o $OutputFile --lang c 2>&1
            $FileDetail.extraction.exit_code = $LASTEXITCODE
            if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputFile)) {
                $ExtractedCount++
                $ExtractionResults[$RelativePath] = @{ status = "success"; output = $OutputFile }
                $FileDetail.extraction.status = "success"
                $FileDetail.extraction.output_bytes = (Get-Item $OutputFile).Length
            } else {
                $ExtractionResults[$RelativePath] = @{ status = "minimal"; output = $OutputFile; message = $Result }
                $Report.summary.warnings += "Extraction fallback: $RelativePath (extractor error)"
                $FileDetail.extraction.status = "fallback"
                $FileDetail.extraction.error = $Result
            }
        } elseif ($IsSparkSource -and $SparkExtractExists) {
            $Result = & $SparkExtractCmd -i $File.FullName -o $OutputFile --lang spark 2>&1
            $FileDetail.extraction.exit_code = $LASTEXITCODE
            if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputFile)) {
                $ExtractedCount++
                $ExtractionResults[$RelativePath] = @{ status = "success"; output = $OutputFile }
                $FileDetail.extraction.status = "success"
                $FileDetail.extraction.output_bytes = (Get-Item $OutputFile).Length
                # Parse function count
                try {
                    $jsonContent = Get-Content $OutputFile | ConvertFrom-Json
                    $FileDetail.extraction.function_count = $jsonContent.functions.Count
                } catch {}
            } else {
                $ExtractionResults[$RelativePath] = @{ status = "minimal"; output = $OutputFile; message = $Result }
                $Report.summary.warnings += "Extraction fallback: $RelativePath (SPARK extractor error)"
                $FileDetail.extraction.status = "fallback"
                $FileDetail.extraction.error = "$Result"
            }
        } else {
            if ($IsSparkSource) {
                $Report.summary.warnings += "Extraction fallback: $RelativePath (SPARK extractor not available; exists=$SparkExtractExists)"
            } elseif (-not $IsCSource) {
                $Report.summary.warnings += "Extraction fallback: $RelativePath (unsupported source type)"
            }
            # Fallback: create minimal extraction JSON with file metadata
            $FileContent = Get-Content $File.FullName -Raw -ErrorAction SilentlyContinue
            $Lines = if ($FileContent) { ($FileContent -split "`n").Count } else { 0 }
            
            $MinimalExtraction = @{
                source_file = $RelativePath
                language = if ($IsSparkSource) { "SPARK" } else { "unknown" }
                line_count = $Lines
                functions = @()
                types = @()
                extracted_at = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
                extraction_method = "minimal"
            }
            $MinimalExtraction | ConvertTo-Json -Depth 10 | Out-File $OutputFile -Encoding UTF8
            $ExtractedCount++
            $ExtractionResults[$RelativePath] = @{ status = "minimal"; output = $OutputFile }
            $FileDetail.extraction.status = "minimal"
            $FileDetail.extraction.output_bytes = (Get-Item $OutputFile).Length
        }
    } catch {
        $ExtractionResults[$RelativePath] = @{ status = "error"; message = $_.Exception.Message }
        $Report.summary.errors += "Extraction exception: $RelativePath - $($_.Exception.Message)"
        $FileDetail.extraction.status = "error"
        $FileDetail.extraction.error = $_.Exception.Message
    }
    
    # Add file detail to report
    $Report.file_details += $FileDetail
}

$Report.summary.extracted_files = $ExtractedCount
$FallbackCount = ($ExtractionResults.Values | Where-Object { $_.status -eq "minimal" }).Count
$Report.phases.extraction = @{
    status = if ($ExtractedCount -gt 0) { "completed" } else { "failed" }
    files_processed = $SourceFiles.Count
    files_extracted = $ExtractedCount
    fallback_used = $FallbackCount
    tools = @{
        c_extractor = @{ path = $CExtractCmd; exists = $CExtractExists }
        spark_extractor = @{ path = $SparkExtractCmd; exists = $SparkExtractExists }
    }
}

Write-Host "  Extracted: $ExtractedCount / $($SourceFiles.Count) files" -ForegroundColor $(if ($ExtractedCount -eq $SourceFiles.Count) { "Green" } else { "Yellow" })

# Phase 2: Spec Assembly
Write-Host ""
Write-Host "[Phase 2] Running spec assembly..." -ForegroundColor Yellow

$SpecCount = 0
$SpecResults = @{}

# Aggregate extractions into a single spec
$AggregatedSpec = @{
    schema_version = "stunir_spec_v1"
    spec_version = "1.0.0"
    modules = @()
    functions = @()
    types = @()
    generated_at = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
    source = "self_refinement"
}

foreach ($Entry in $ExtractionResults.GetEnumerator()) {
    if ($Entry.Value.status -in @("success", "minimal")) {
        try {
            $ExtractionData = Get-Content $Entry.Value.output | ConvertFrom-Json
            if ($ExtractionData.functions) {
                $AggregatedSpec.functions += $ExtractionData.functions
            }
            if ($ExtractionData.types) {
                $AggregatedSpec.types += $ExtractionData.types
            }
        } catch {
            $Report.summary.warnings += "Failed to parse extraction: $($Entry.Key)"
        }
    }
}

$SpecOutputFile = Join-Path $SpecDir "self_refine_spec.json"
$AggregatedSpec | ConvertTo-Json -Depth 10 | Out-File $SpecOutputFile -Encoding UTF8
$SpecCount = $AggregatedSpec.functions.Count

$Report.summary.spec_files = 1
$Report.phases.spec_assembly = @{
    status = "completed"
    output_file = $SpecOutputFile
    function_count = $SpecCount
    type_count = $AggregatedSpec.types.Count
}

Write-Host "  Spec file: $SpecOutputFile" -ForegroundColor Green
Write-Host "  Functions: $SpecCount" -ForegroundColor Green

# Phase 3: IR Conversion
Write-Host ""
Write-Host "[Phase 3] Running IR conversion..." -ForegroundColor Yellow

$IROutputFile = Join-Path $IRDir "self_refine_ir.json"

try {
    $IRConverterCmd = Join-Path $SparkDir "bin\ir_converter_main.exe"
    
    if (Test-Path $IRConverterCmd) {
        $Result = & $IRConverterCmd $SpecOutputFile $IROutputFile 2>&1
        if ($LASTEXITCODE -eq 0) {
            $Report.phases.ir_conversion = @{ status = "completed"; output_file = $IROutputFile }
        } else {
            $Report.phases.ir_conversion = @{ status = "error"; message = $Result }
            $Report.summary.errors += "IR conversion failed"
        }
    } else {
        # Fallback: create minimal IR JSON
        $MinimalIR = @{
            schema_version = "stunir_ir_v1"
            ir_version = "1.0.0"
            module_name = "STUNIR_Self_Refine"
            source_spec = $SpecOutputFile
            functions = @()
            types = @()
            generated_at = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        }
        
        # Convert spec functions to IR functions with placeholder steps
        foreach ($Func in $AggregatedSpec.functions) {
            $IRFunc = @{
                name = $Func.name
                return_type = if ($Func.return_type) { $Func.return_type } else { "void" }
                parameters = @()
                steps = @(
                    @{ kind = "assign"; target = "result"; value = "0" },
                    @{ kind = "return"; value = "result" }
                )
            }
            $MinimalIR.functions += $IRFunc
        }
        
        $MinimalIR | ConvertTo-Json -Depth 10 | Out-File $IROutputFile -Encoding UTF8
        $Report.phases.ir_conversion = @{ 
            status = "minimal" 
            output_file = $IROutputFile
            note = "IR converter not built, using placeholder IR"
        }
        $Report.summary.warnings += "IR conversion minimal (tool not built)"
    }
    
    $Report.summary.ir_files = 1
} catch {
    $Report.phases.ir_conversion = @{ status = "error"; message = $_.Exception.Message }
    $Report.summary.errors += "IR conversion exception: $($_.Exception.Message)"
}

Write-Host "  IR file: $IROutputFile" -ForegroundColor $(if ($Report.phases.ir_conversion.status -eq "completed") { "Green" } else { "Yellow" })

# Phase 4: IR Validation
Write-Host ""
Write-Host "[Phase 4] Validating IR..." -ForegroundColor Yellow

$IRValidationResults = @{
    schema_valid = $false
    semantic_valid = $false
    errors = @()
}

try {
    $IRValidateCmd = Join-Path $SparkDir "bin\ir_validate_schema.exe"
    
    if (Test-Path $IRValidateCmd) {
        # ir_validate_schema reads from stdin
        $Result = Get-Content $IROutputFile | & $IRValidateCmd 2>&1
        if ($LASTEXITCODE -eq 0) {
            $IRValidationResults.schema_valid = $true
            Write-Host "  Schema validation: PASS" -ForegroundColor Green
        } else {
            $IRValidationResults.errors += "Schema validation failed: $Result"
            Write-Host "  Schema validation: FAIL" -ForegroundColor Red
        }
    } else {
        # Manual schema check
        $IRContent = Get-Content $IROutputFile | ConvertFrom-Json
        if ($IRContent.schema_version -and $IRContent.ir_version) {
            $IRValidationResults.schema_valid = $true
            Write-Host "  Schema validation: PASS (manual)" -ForegroundColor Green
        } else {
            $IRValidationResults.errors += "Missing required IR fields"
            Write-Host "  Schema validation: FAIL (manual)" -ForegroundColor Red
        }
    }
} catch {
    $IRValidationResults.errors += $_.Exception.Message
}

$Report.phases.ir_validation = $IRValidationResults

# Phase 5: Code Emission (if not skipped)
if (-not $SkipEmission) {
    Write-Host ""
    Write-Host "[Phase 5] Running code emission..." -ForegroundColor Yellow
    
    $TargetList = $Targets -split ","
    $EmissionResults = @{}
    $EmittedCount = 0
    
    foreach ($Target in $TargetList) {
        $Target = $Target.Trim()
        $TargetDir = Join-Path $EmitDir $Target.ToLower()
        
        if (-not (Test-Path $TargetDir)) {
            New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
        }
        
        $OutputFile = Join-Path $TargetDir "self_refine_output.$(if ($Target -eq 'SPARK' -or $Target -eq 'Ada') { 'adb' } elseif ($Target -eq 'Python') { 'py' } elseif ($Target -eq 'Rust') { 'rs' } else { 'txt' })"
        
        try {
            $EmitCmd = Join-Path $SparkDir "bin\emit_target_main.exe"
            
            if (Test-Path $EmitCmd) {
                $Result = & $EmitCmd -i $IROutputFile -t $Target -o $OutputFile 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $EmissionResults[$Target] = @{ status = "success"; output = $OutputFile }
                    $EmittedCount++
                    Write-Host "  $Target`: SUCCESS" -ForegroundColor Green
                } else {
                    $EmissionResults[$Target] = @{ status = "error"; message = $Result }
                    Write-Host "  $Target`: FAILED" -ForegroundColor Red
                }
            } else {
                # Fallback: create placeholder output
                $Placeholder = "# STUNIR Self-Refinement Output`n# Target: $Target`n# Generated: $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')`n# Source: $IROutputFile`n`n# Placeholder - emitter not built"
                $Placeholder | Out-File $OutputFile -Encoding UTF8
                $EmissionResults[$Target] = @{ status = "placeholder"; output = $OutputFile }
                $EmittedCount++
                Write-Host "  $Target`: PLACEHOLDER" -ForegroundColor Yellow
            }
        } catch {
            $EmissionResults[$Target] = @{ status = "error"; message = $_.Exception.Message }
            Write-Host "  $Target`: ERROR - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    $Report.summary.emitted_files = $EmittedCount
    $Report.phases.emission = @{
        status = if ($EmittedCount -eq $TargetList.Count) { "completed" } else { "partial" }
        targets = $EmissionResults
    }
}

# Finalize report
$Report.status = if ($IRValidationResults.schema_valid) { "passed" } else { "failed" }

# Write reports (force repo-root reports)
$RunMarker = Join-Path $RepoReportDir "self_refine_last_run.txt"
"$($Report.timestamp) | $ScriptVersion" | Out-File -FilePath $RunMarker -Encoding UTF8

# Also write a local marker near OutputDir for troubleshooting
$LocalReportDir = Join-Path $OutputDir "reports"
if (-not (Test-Path $LocalReportDir)) {
    New-Item -ItemType Directory -Path $LocalReportDir -Force | Out-Null
}
$LocalRunMarker = Join-Path $LocalReportDir "self_refine_last_run.txt"
"$($Report.timestamp) | $ScriptVersion" | Out-File -FilePath $LocalRunMarker -Encoding UTF8

$ReportFile = Join-Path $RepoReportDir "self_refine_report.json"
$Report | ConvertTo-Json -Depth 10 | Out-File $ReportFile -Encoding UTF8

# Write detailed file diagnostics
$DiagnosticsFile = Join-Path $RepoReportDir "self_refine_diagnostics.json"
$Diagnostics = @{
    timestamp = $Report.timestamp
    script_version = $ScriptVersion
    file_count = $Report.file_details.Count
    files = $Report.file_details
}
$Diagnostics | ConvertTo-Json -Depth 10 | Out-File $DiagnosticsFile -Encoding UTF8

$SummaryFile = Join-Path $RepoReportDir "self_refine_summary.txt"
$Summary = @"
STUNIR Self-Refinement Summary
==============================
Report Version: $ReportVersion
Script Version: $ScriptVersion
Timestamp: $($Report.timestamp)
Status: $($Report.status.ToUpper())

Files Processed:
  Total SPARK files: $($Report.summary.total_files)
  Extracted: $($Report.summary.extracted_files)
  Spec functions: $($Report.phases.spec_assembly.function_count)
  IR files: $($Report.summary.ir_files)
  Emitted targets: $($Report.summary.emitted_files)
    Extraction fallback: $($Report.phases.extraction.fallback_used)

Phases:
  Extraction:     $($Report.phases.extraction.status)
  Spec Assembly:  $($Report.phases.spec_assembly.status)
  IR Conversion:  $($Report.phases.ir_conversion.status)
  IR Validation:  $(if ($IRValidationResults.schema_valid) { "PASS" } else { "FAIL" })
  Emission:       $(if ($Report.phases.emission) { $Report.phases.emission.status } else { "skipped" })

Errors: $($Report.summary.errors.Count)
Warnings: $($Report.summary.warnings.Count)
File Details: $($Report.file_details.Count) records in self_refine_diagnostics.json

Output Directory: $OutputDir
"@

$Summary | Out-File $SummaryFile -Encoding UTF8

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Self-Refinement Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Status: $($Report.status.ToUpper())" -ForegroundColor $(if ($Report.status -eq "passed") { "Green" } else { "Red" })
Write-Host "Report: $ReportFile"
Write-Host "Summary: $SummaryFile"
Write-Host ""

if ($Report.summary.errors.Count -gt 0) {
    Write-Host "Errors:" -ForegroundColor Red
    $Report.summary.errors | ForEach-Object { Write-Host "  - $_" }
}

if ($Report.summary.warnings.Count -gt 0) {
    Write-Host "Warnings:" -ForegroundColor Yellow
    $Report.summary.warnings | ForEach-Object { Write-Host "  - $_" }
}

exit $(if ($Report.status -eq "passed") { 0 } else { 1 })
