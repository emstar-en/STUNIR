# Test Python Pipeline Script for Windows
# Tests spec_to_ir.py -> ir_to_code.py pipeline

$ErrorActionPreference = "Stop"

Write-Host "=== STUNIR Python Pipeline Test ===" -ForegroundColor Cyan

# Check Python is available
try {
    $pyVersion = python --version 2>&1
    Write-Host "Python version: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Create test directories
$testDir = "test_output"
New-Item -ItemType Directory -Force -Path $testDir | Out-Null

# Test 1: spec_to_ir.py
Write-Host "`n[Test 1] Running spec_to_ir.py..." -ForegroundColor Yellow
$specFile = "test_python_pipeline/specs/simple_test.json"
$irOutput = "$testDir/simple_test_ir.json"

python tools/spec_to_ir.py --input $specFile --output $irOutput --emit-attestation 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: spec_to_ir.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $irOutput)) {
    Write-Host "FAILED: IR output file not created" -ForegroundColor Red
    exit 1
}

Write-Host "PASSED: spec_to_ir.py" -ForegroundColor Green

# Test 2: ir_to_code.py (Python output)
Write-Host "`n[Test 2] Running ir_to_code.py (Python)..." -ForegroundColor Yellow
$codeOutput = "$testDir/simple_test.py"

python tools/ir_to_code.py --input $irOutput --output $codeOutput --target python --emit-attestation 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: ir_to_code.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $codeOutput)) {
    Write-Host "FAILED: Code output file not created" -ForegroundColor Red
    exit 1
}

Write-Host "PASSED: ir_to_code.py (Python)" -ForegroundColor Green

# Test 3: Verify generated Python code is valid
Write-Host "`n[Test 3] Verifying generated Python code..." -ForegroundColor Yellow
try {
    python -m py_compile $codeOutput 2>&1
    Write-Host "PASSED: Generated Python code is valid" -ForegroundColor Green
} catch {
    Write-Host "FAILED: Generated Python code has syntax errors" -ForegroundColor Red
    exit 1
}

# Display generated code
Write-Host "`n=== Generated Python Code ===" -ForegroundColor Cyan
Get-Content $codeOutput | ForEach-Object { Write-Host $_ }

Write-Host "`n=== All Tests PASSED ===" -ForegroundColor Green
Write-Host "Output files in: $testDir/" -ForegroundColor Gray
