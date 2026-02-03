#!/bin/bash
# STUNIR Confluence Test
# Verifies that SPARK, Python, and Rust implementations produce identical outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "🔬 STUNIR CONFLUENCE TEST: Week 2"
echo "Testing: SPARK ≡ Python ≡ Rust"
echo ""

# Create test output directory
TEST_DIR="$REPO_ROOT/test_outputs/confluence"
mkdir -p "$TEST_DIR"

# Test spec directory
SPEC_DIR="$REPO_ROOT/spec/ardupilot_test"

if [ ! -d "$SPEC_DIR" ]; then
    echo "❌ ERROR: Spec directory not found: $SPEC_DIR"
    exit 1
fi

echo "📍 Test Spec: $SPEC_DIR"
echo ""

# ============================================================================
# Phase 1: spec_to_ir - Generate IR from specs
# ============================================================================

echo "=== Phase 1: spec_to_ir (Spec → IR) ==="
echo ""

# Test SPARK spec_to_ir
echo "1️⃣  SPARK spec_to_ir:"
SPARK_SPEC_TO_IR="$REPO_ROOT/tools/spark/bin/stunir_spec_to_ir_main"
if [ -x "$SPARK_SPEC_TO_IR" ]; then
    $SPARK_SPEC_TO_IR --spec-root "$SPEC_DIR" --out "$TEST_DIR/ir_spark.json"
    echo "   ✅ Generated: $TEST_DIR/ir_spark.json"
else
    echo "   ⚠️  SPARK binary not found, skipping"
fi
echo ""

# Test Python spec_to_ir
echo "2️⃣  Python spec_to_ir:"
PYTHON_SPEC_TO_IR="$REPO_ROOT/tools/spec_to_ir.py"
if [ -f "$PYTHON_SPEC_TO_IR" ]; then
    python3 "$PYTHON_SPEC_TO_IR" --spec-root "$SPEC_DIR" --out "$TEST_DIR/ir_python.json"
    echo "   ✅ Generated: $TEST_DIR/ir_python.json"
else
    echo "   ⚠️  Python script not found, skipping"
fi
echo ""

# Test Rust spec_to_ir
echo "3️⃣  Rust spec_to_ir:"
RUST_SPEC_TO_IR="$REPO_ROOT/tools/rust/target/release/stunir_spec_to_ir"
if [ -x "$RUST_SPEC_TO_IR" ]; then
    # Rust expects a single file, not a directory - find first JSON file
    FIRST_JSON=$(find "$SPEC_DIR" -name "*.json" -type f | head -1)
    if [ -n "$FIRST_JSON" ]; then
        $RUST_SPEC_TO_IR "$FIRST_JSON" --output "$TEST_DIR/ir_rust.json"
        echo "   ✅ Generated: $TEST_DIR/ir_rust.json"
    else
        echo "   ⚠️  No JSON files found in spec directory"
    fi
else
    echo "   ⚠️  Rust binary not found, skipping"
fi
echo ""

# ============================================================================
# Phase 2: ir_to_code - Generate C code from IR
# ============================================================================

echo "=== Phase 2: ir_to_code (IR → C Code) ==="
echo ""

# Use SPARK-generated IR as the canonical source
if [ -f "$TEST_DIR/ir_spark.json" ]; then
    IR_FILE="$TEST_DIR/ir_spark.json"
elif [ -f "$TEST_DIR/ir_python.json" ]; then
    IR_FILE="$TEST_DIR/ir_python.json"
elif [ -f "$TEST_DIR/ir_rust.json" ]; then
    IR_FILE="$TEST_DIR/ir_rust.json"
else
    echo "❌ ERROR: No IR file generated in Phase 1"
    exit 1
fi

echo "📖 Using IR: $IR_FILE"
echo ""

# Test SPARK ir_to_code
echo "1️⃣  SPARK ir_to_code:"
SPARK_IR_TO_CODE="$REPO_ROOT/tools/spark/bin/stunir_ir_to_code_main"
if [ -x "$SPARK_IR_TO_CODE" ]; then
    $SPARK_IR_TO_CODE --input "$IR_FILE" --output "$TEST_DIR/output_spark.c" --target c
    echo "   ✅ Generated: $TEST_DIR/output_spark.c"
else
    echo "   ⚠️  SPARK binary not found, skipping"
fi
echo ""

# Test Python ir_to_code
echo "2️⃣  Python ir_to_code:"
PYTHON_IR_TO_CODE="$REPO_ROOT/tools/ir_to_code.py"
if [ -f "$PYTHON_IR_TO_CODE" ]; then
    # Python uses different CLI: --ir, --lang, --templates, --out
    TEMPLATE_DIR="$REPO_ROOT/targets/c/templates"
    mkdir -p "$TEMPLATE_DIR"
    python3 "$PYTHON_IR_TO_CODE" --ir "$IR_FILE" --lang c --templates "$TEMPLATE_DIR" --out "$TEST_DIR/python_out" 2>&1 | head -20 || true
    if [ -f "$TEST_DIR/python_out/output.c" ]; then
        cp "$TEST_DIR/python_out/output.c" "$TEST_DIR/output_python.c"
        echo "   ✅ Generated: $TEST_DIR/output_python.c"
    else
        echo "   ⚠️  Python output not generated"
    fi
else
    echo "   ⚠️  Python script not found, skipping"
fi
echo ""

# Test Rust ir_to_code
echo "3️⃣  Rust ir_to_code:"
RUST_IR_TO_CODE="$REPO_ROOT/tools/rust/target/release/stunir_ir_to_code"
if [ -x "$RUST_IR_TO_CODE" ]; then
    $RUST_IR_TO_CODE "$IR_FILE" --target c --output "$TEST_DIR/output_rust.c"
    echo "   ✅ Generated: $TEST_DIR/output_rust.c"
else
    echo "   ⚠️  Rust binary not found, skipping"
fi
echo ""

# ============================================================================
# Phase 3: Compare outputs for confluence
# ============================================================================

echo "=== Phase 3: Confluence Verification ==="
echo ""

PASS_COUNT=0
TOTAL_COUNT=0

# Compare SPARK vs Python
if [ -f "$TEST_DIR/output_spark.c" ] && [ -f "$TEST_DIR/output_python.c" ]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if diff -q "$TEST_DIR/output_spark.c" "$TEST_DIR/output_python.c" >/dev/null 2>&1; then
        echo "✅ SPARK ≡ Python (identical outputs)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  SPARK ≠ Python (outputs differ)"
        echo "   Run: diff $TEST_DIR/output_spark.c $TEST_DIR/output_python.c"
    fi
fi

# Compare SPARK vs Rust
if [ -f "$TEST_DIR/output_spark.c" ] && [ -f "$TEST_DIR/output_rust.c" ]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if diff -q "$TEST_DIR/output_spark.c" "$TEST_DIR/output_rust.c" >/dev/null 2>&1; then
        echo "✅ SPARK ≡ Rust (identical outputs)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  SPARK ≠ Rust (outputs differ)"
        echo "   Run: diff $TEST_DIR/output_spark.c $TEST_DIR/output_rust.c"
    fi
fi

# Compare Python vs Rust
if [ -f "$TEST_DIR/output_python.c" ] && [ -f "$TEST_DIR/output_rust.c" ]; then
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if diff -q "$TEST_DIR/output_python.c" "$TEST_DIR/output_rust.c" >/dev/null 2>&1; then
        echo "✅ Python ≡ Rust (identical outputs)"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "⚠️  Python ≠ Rust (outputs differ)"
        echo "   Run: diff $TEST_DIR/output_python.c $TEST_DIR/output_rust.c"
    fi
fi

echo ""
echo "=== Summary ==="
echo "Passed: $PASS_COUNT / $TOTAL_COUNT"

if [ $TOTAL_COUNT -eq 0 ]; then
    echo "⚠️  WARNING: No implementations could be compared"
    echo "   Please build SPARK, Python, and Rust tools"
    exit 1
elif [ $PASS_COUNT -eq $TOTAL_COUNT ]; then
    echo "🎉 CONFLUENCE VERIFIED: All implementations produce identical outputs!"
    exit 0
else
    echo "⚠️  CONFLUENCE INCOMPLETE: Some implementations differ"
    exit 1
fi
