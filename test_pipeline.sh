#!/usr/bin/env bash
# Test Python Pipeline Script for Linux/WSL
# Tests spec_to_ir.py -> ir_to_code.py pipeline

set -euo pipefail

echo "=== STUNIR Python Pipeline Test ==="

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found in PATH"
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1)
echo "Python version: $PY_VERSION"

# Create test directories
TEST_DIR="test_output"
mkdir -p "$TEST_DIR"

# Test 1: spec_to_ir.py
echo ""
echo "[Test 1] Running spec_to_ir.py..."
SPEC_FILE="test_python_pipeline/specs/simple_test.json"
IR_OUTPUT="$TEST_DIR/simple_test_ir.json"

if ! python3 tools/spec_to_ir.py --input "$SPEC_FILE" --output "$IR_OUTPUT" --emit-attestation 2>&1; then
    echo "FAILED: spec_to_ir.py exited with code $?"
    exit 1
fi

if [ ! -f "$IR_OUTPUT" ]; then
    echo "FAILED: IR output file not created"
    exit 1
fi

echo "PASSED: spec_to_ir.py"

# Test 2: ir_to_code.py (Python output)
echo ""
echo "[Test 2] Running ir_to_code.py (Python)..."
CODE_OUTPUT="$TEST_DIR/simple_test.py"

if ! python3 tools/ir_to_code.py --input "$IR_OUTPUT" --output "$CODE_OUTPUT" --target python --emit-attestation 2>&1; then
    echo "FAILED: ir_to_code.py exited with code $?"
    exit 1
fi

if [ ! -f "$CODE_OUTPUT" ]; then
    echo "FAILED: Code output file not created"
    exit 1
fi

echo "PASSED: ir_to_code.py (Python)"

# Test 3: Verify generated Python code is valid
echo ""
echo "[Test 3] Verifying generated Python code..."
if python3 -m py_compile "$CODE_OUTPUT" 2>&1; then
    echo "PASSED: Generated Python code is valid"
else
    echo "FAILED: Generated Python code has syntax errors"
    exit 1
fi

# Display generated code
echo ""
echo "=== Generated Python Code ==="
cat "$CODE_OUTPUT"

echo ""
echo "=== All Tests PASSED ==="
echo "Output files in: $TEST_DIR/"
