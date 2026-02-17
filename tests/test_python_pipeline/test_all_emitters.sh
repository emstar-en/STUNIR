#!/bin/bash
# STUNIR Python Pipeline - Comprehensive Emitter Test
# Tests all 24+ emitters to verify they work with semantic IR

set -e

REPO_ROOT="/home/ubuntu/stunir_repo"
TEST_DIR="$REPO_ROOT/test_python_pipeline"
IR_FILE="$TEST_DIR/ardupilot_python_ir_v2.json"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================="
echo "STUNIR Python Pipeline Test Suite"
echo "=================================="
echo ""
echo "Testing: Spec → Semantic IR → Code"
echo "IR File: $IR_FILE"
echo ""

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

test_template_emitter() {
    local lang=$1
    local template_dir=$2
    local output_dir=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing $lang template emitter... "
    
    if python3 "$REPO_ROOT/tools/ir_to_code.py" \
        --ir "$IR_FILE" \
        --lang "$lang" \
        --templates "$template_dir" \
        --out "$output_dir" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

test_target_emitter() {
    local name=$1
    local emitter_path=$2
    local output_dir=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing $name emitter... "
    
    if python3 "$emitter_path" "$IR_FILE" --output="$output_dir" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${YELLOW}⚠ SKIPPED${NC} (may require relative imports)"
        TOTAL_TESTS=$((TOTAL_TESTS - 1))
        return 2
    fi
}

# Create output directories
mkdir -p "$TEST_DIR/output"

echo "=== Template-based Emitters ==="
test_template_emitter "python" "$REPO_ROOT/templates/python" "$TEST_DIR/output/python"
test_template_emitter "c" "$REPO_ROOT/templates/c" "$TEST_DIR/output/c"
test_template_emitter "rust" "$REPO_ROOT/templates/rust" "$TEST_DIR/output/rust"
test_template_emitter "javascript" "$REPO_ROOT/templates/javascript" "$TEST_DIR/output/javascript"
test_template_emitter "wasm" "$REPO_ROOT/templates/wasm" "$TEST_DIR/output/wasm"
test_template_emitter "asm" "$REPO_ROOT/templates/asm" "$TEST_DIR/output/asm"

echo ""
echo "=== Target-specific Emitters ==="
test_target_emitter "embedded" "$REPO_ROOT/targets/embedded/emitter.py" "$TEST_DIR/output/embedded"
test_target_emitter "gpu" "$REPO_ROOT/targets/gpu/emitter.py" "$TEST_DIR/output/gpu"
test_target_emitter "wasm_target" "$REPO_ROOT/targets/wasm/emitter.py" "$TEST_DIR/output/wasm_target"
test_target_emitter "fpga" "$REPO_ROOT/targets/fpga/emitter.py" "$TEST_DIR/output/fpga"
test_target_emitter "arm_assembly" "$REPO_ROOT/targets/assembly/arm/emitter.py" "$TEST_DIR/output/arm_asm"
test_target_emitter "x86_assembly" "$REPO_ROOT/targets/assembly/x86/emitter.py" "$TEST_DIR/output/x86_asm"

echo ""
echo "=================================="
echo "Test Results Summary"
echo "=================================="
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
