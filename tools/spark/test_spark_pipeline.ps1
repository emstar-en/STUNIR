#!/usr/bin/env pwsh
# STUNIR SPARK Pipeline E2E Test Script
# Tests the complete SPARK-only pipeline from IR validation to code emission
# Copyright (C) 2026 STUNIR Project
# SPDX-License-Identifier: Apache-2.0

param(
    [string]$TestIR = "test\ir_tools\test_control_flow_ir.json",
    [string]$OutputDir = "test_pipeline_output",
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

Write-Host "=== STUNIR SPARK Pipeline E2E Test ===" -ForegroundColor Cyan
Write-Host "Test IR: $TestIR"
Write-Host "Output:  $OutputDir"
Write-Host ""

# Clean output directory
if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

# Step 1: Validate IR
Write-Host "[1/4] Validating IR..." -ForegroundColor Yellow
$ValidateResult = Get-Content $TestIR | bin\ir_validate_schema.exe 2>&1
if ($ValidateResult -ne "true") {
    Write-Host "FAILED: IR validation failed" -ForegroundColor Red
    Write-Host $ValidateResult
    exit 1
}
Write-Host "      IR validation: PASSED" -ForegroundColor Green

# Step 2: Emit C99
Write-Host "[2/4] Emitting C99..." -ForegroundColor Yellow
$COutput = Join-Path $OutputDir "output.c"
bin\emit_target_main.exe -i $TestIR -o $COutput -t c99 2>&1 | Out-Null
if (-not (Test-Path $COutput)) {
    Write-Host "FAILED: C99 emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      C99 emission:  PASSED" -ForegroundColor Green

# Step 3: Emit Clojure
Write-Host "[3/4] Emitting Clojure..." -ForegroundColor Yellow
$CljOutput = Join-Path $OutputDir "output.clj"
bin\emit_target_main.exe -i $TestIR -o $CljOutput -t clojure 2>&1 | Out-Null
if (-not (Test-Path $CljOutput)) {
    Write-Host "FAILED: Clojure emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Clojure emission: PASSED" -ForegroundColor Green

# Step 4: Emit Futhark and Lean4
Write-Host "[4/11] Emitting Futhark and Lean4..." -ForegroundColor Yellow
$FutOutput = Join-Path $OutputDir "output.fut"
$LeanOutput = Join-Path $OutputDir "output.lean"
bin\emit_target_main.exe -i $TestIR -o $FutOutput -t futhark 2>&1 | Out-Null
bin\emit_target_main.exe -i $TestIR -o $LeanOutput -t lean4 2>&1 | Out-Null
if (-not (Test-Path $FutOutput) -or -not (Test-Path $LeanOutput)) {
    Write-Host "FAILED: Futhark/Lean4 emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Futhark emission: PASSED" -ForegroundColor Green
Write-Host "      Lean4 emission:   PASSED" -ForegroundColor Green

# Step 5: Emit Lisp family (Common Lisp, Scheme)
Write-Host "[5/11] Emitting Lisp family..." -ForegroundColor Yellow
$ClOutput = Join-Path $OutputDir "output.lisp"
$ScmOutput = Join-Path $OutputDir "output.scm"
bin\emit_target_main.exe -i $TestIR -o $ClOutput -t common-lisp 2>&1 | Out-Null
bin\emit_target_main.exe -i $TestIR -o $ScmOutput -t scheme 2>&1 | Out-Null
if (-not (Test-Path $ClOutput) -or -not (Test-Path $ScmOutput)) {
    Write-Host "FAILED: Lisp family emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Common Lisp emission: PASSED" -ForegroundColor Green
Write-Host "      Scheme emission:      PASSED" -ForegroundColor Green

# Step 6: Emit Prolog family (SWI-Prolog)
Write-Host "[6/11] Emitting Prolog family..." -ForegroundColor Yellow
$PlOutput = Join-Path $OutputDir "output.pl"
bin\emit_target_main.exe -i $TestIR -o $PlOutput -t swi-prolog 2>&1 | Out-Null
if (-not (Test-Path $PlOutput)) {
    Write-Host "FAILED: Prolog emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      SWI-Prolog emission:  PASSED" -ForegroundColor Green

# Step 7: Emit Haskell
Write-Host "[7/11] Emitting Haskell..." -ForegroundColor Yellow
$HsOutput = Join-Path $OutputDir "output.hs"
bin\emit_target_main.exe -i $TestIR -o $HsOutput -t haskell 2>&1 | Out-Null
if (-not (Test-Path $HsOutput)) {
    Write-Host "FAILED: Haskell emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Haskell emission:    PASSED" -ForegroundColor Green

# Step 8: Emit Rust
Write-Host "[8/11] Emitting Rust..." -ForegroundColor Yellow
$RsOutput = Join-Path $OutputDir "output.rs"
bin\emit_target_main.exe -i $TestIR -o $RsOutput -t rust 2>&1 | Out-Null
if (-not (Test-Path $RsOutput)) {
    Write-Host "FAILED: Rust emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Rust emission:       PASSED" -ForegroundColor Green

# Step 9: Emit Python
Write-Host "[9/11] Emitting Python..." -ForegroundColor Yellow
$PyOutput = Join-Path $OutputDir "output.py"
bin\emit_target_main.exe -i $TestIR -o $PyOutput -t python 2>&1 | Out-Null
if (-not (Test-Path $PyOutput)) {
    Write-Host "FAILED: Python emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Python emission:     PASSED" -ForegroundColor Green

# Step 10: Emit SPARK
Write-Host "[10/11] Emitting SPARK..." -ForegroundColor Yellow
$SparkOutput = Join-Path $OutputDir "output_spark.adb"
bin\emit_target_main.exe -i $TestIR -o $SparkOutput -t spark 2>&1 | Out-Null
if (-not (Test-Path $SparkOutput)) {
    Write-Host "FAILED: SPARK emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      SPARK emission:      PASSED" -ForegroundColor Green

# Step 11: Emit Ada
Write-Host "[11/11] Emitting Ada..." -ForegroundColor Yellow
$AdaOutput = Join-Path $OutputDir "output_ada.adb"
bin\emit_target_main.exe -i $TestIR -o $AdaOutput -t ada 2>&1 | Out-Null
if (-not (Test-Path $AdaOutput)) {
    Write-Host "FAILED: Ada emission failed" -ForegroundColor Red
    exit 1
}
Write-Host "      Ada emission:        PASSED" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "=== Pipeline Test Summary ===" -ForegroundColor Cyan
Write-Host "All 11 stages passed successfully!"
Write-Host ""
Write-Host "Generated files:"
Get-ChildItem $OutputDir | ForEach-Object {
    Write-Host "  $($_.Name) ($($_.Length) bytes)"
}

if ($Verbose) {
    Write-Host ""
    Write-Host "=== C99 Output ===" -ForegroundColor Cyan
    Get-Content $COutput
    Write-Host ""
    Write-Host "=== Clojure Output ===" -ForegroundColor Cyan
    Get-Content $CljOutput
    Write-Host ""
    Write-Host "=== Haskell Output ===" -ForegroundColor Cyan
    Get-Content $HsOutput
    Write-Host ""
    Write-Host "=== Rust Output ===" -ForegroundColor Cyan
    Get-Content $RsOutput
    Write-Host ""
    Write-Host "=== Python Output ===" -ForegroundColor Cyan
    Get-Content $PyOutput
    Write-Host ""
    Write-Host "=== SPARK Output ===" -ForegroundColor Cyan
    Get-Content $SparkOutput
    Write-Host ""
    Write-Host "=== Ada Output ===" -ForegroundColor Cyan
    Get-Content $AdaOutput
}

Pop-Location
Write-Host ""
Write-Host "E2E test completed successfully." -ForegroundColor Green