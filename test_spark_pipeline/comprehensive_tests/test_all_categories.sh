#!/bin/bash
# STUNIR SPARK Pipeline Comprehensive Test Suite
# Tests spec_to_ir -> ir_to_code for multiple categories

set -e

REPO_ROOT="/home/ubuntu/stunir_repo"
TEST_DIR="$REPO_ROOT/test_spark_pipeline/comprehensive_tests"
SPEC_TO_IR="$REPO_ROOT/tools/spark/bin/stunir_spec_to_ir_main"
IR_TO_CODE="$REPO_ROOT/tools/spark/bin/stunir_ir_to_code_main"

# Target categories to test (representing the 24+ emitter categories)
TARGETS=(
    "python"
    "rust"
    "c"
    "cpp"
    "go"
    "javascript"
    "typescript"
    "java"
    "csharp"
)

echo "[TEST] STUNIR SPARK Pipeline Comprehensive Test Suite"
echo "======================================================"
echo ""

PASSED=0
FAILED=0

for target in "${TARGETS[@]}"; do
    echo "[TEST] Category: $target"
    
    # Generate IR from spec
    if $SPEC_TO_IR \
        --spec-root "$TEST_DIR/../specs" \
        --out "$TEST_DIR/ir_${target}.json" \
        --lockfile "$REPO_ROOT/local_toolchain.lock.json" > /dev/null 2>&1; then
        
        # Verify IR has proper schema
        if grep -q '"schema":"stunir_ir_v1"' "$TEST_DIR/ir_${target}.json"; then
            echo "  ✅ spec_to_ir: Generated proper semantic IR"
            
            # Generate code from IR
            if $IR_TO_CODE \
                --input "$TEST_DIR/ir_${target}.json" \
                --target "$target" \
                --output "$TEST_DIR/output_${target}" 2>&1 | grep -q "Emitted"; then
                
                echo "  ✅ ir_to_code: Generated $target code successfully"
                PASSED=$((PASSED + 1))
            else
                echo "  ❌ ir_to_code: Failed to generate $target code"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "  ❌ spec_to_ir: Did not generate proper semantic IR"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ❌ spec_to_ir: Failed to generate IR"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "======================================================"
echo "[RESULT] SPARK Pipeline Tests:"
echo "  ✅ Passed: $PASSED"
echo "  ❌ Failed: $FAILED"
echo "  Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "[SUCCESS] All SPARK pipeline tests passed!"
    exit 0
else
    echo "[FAILURE] Some tests failed"
    exit 1
fi
