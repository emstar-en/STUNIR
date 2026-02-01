#!/bin/bash
# STUNIR SPARK Test Runner v0.8.6
# Runs all SPARK pipeline tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPARK_TOOLS="$PROJECT_ROOT/tools/spark"
TEST_DIR="$SCRIPT_DIR"
RESULTS_DIR="$TEST_DIR/results"

mkdir -p "$RESULTS_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

log_pass() { echo -e "${GREEN}✓${NC} $1"; }
log_fail() { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${YELLOW}→${NC} $1"; }

echo "========================================"
echo "STUNIR SPARK Test Suite v0.8.6"
echo "========================================"
echo ""

# Check for SPARK tools
SPEC_TO_IR="$SPARK_TOOLS/bin/stunir_spec_to_ir_main"
IR_TO_CODE="$SPARK_TOOLS/bin/stunir_ir_to_code_main"

if [ ! -f "$SPEC_TO_IR" ]; then
    log_info "Building SPARK tools..."
    cd "$SPARK_TOOLS"
    gprbuild -P stunir_tools.gpr 2>/dev/null || {
        log_fail "SPARK tools not available (gprbuild failed)"
        SPEC_TO_IR=""
    }
fi

echo ""
echo "=== UNIT TESTS: spec_to_ir ==="

python3 "$TEST_DIR/unit/test_spec_to_ir.py" "$SPEC_TO_IR" 2>/dev/null && {
    UNIT_RESULT=$(python3 "$TEST_DIR/unit/test_spec_to_ir.py" "$SPEC_TO_IR" 2>&1 | tail -1)
    UNIT_PASSED=$(echo "$UNIT_RESULT" | grep -oP '\d+(?= passed)' || echo "0")
    UNIT_TOTAL=$(echo "$UNIT_RESULT" | grep -oP '\d+(?= total)' || echo "0")
    PASSED_TESTS=$((PASSED_TESTS + UNIT_PASSED))
    TOTAL_TESTS=$((TOTAL_TESTS + UNIT_TOTAL))
} || log_fail "spec_to_ir unit tests"

echo ""
echo "=== UNIT TESTS: ir_to_code ==="

python3 "$TEST_DIR/unit/test_ir_to_code.py" "$IR_TO_CODE" 2>/dev/null && {
    UNIT_RESULT=$(python3 "$TEST_DIR/unit/test_ir_to_code.py" "$IR_TO_CODE" 2>&1 | tail -1)
    UNIT_PASSED=$(echo "$UNIT_RESULT" | grep -oP '\d+(?= passed)' || echo "0")
    UNIT_TOTAL=$(echo "$UNIT_RESULT" | grep -oP '\d+(?= total)' || echo "0")
    PASSED_TESTS=$((PASSED_TESTS + UNIT_PASSED))
    TOTAL_TESTS=$((TOTAL_TESTS + UNIT_TOTAL))
} || log_fail "ir_to_code unit tests"

echo ""
echo "=== INTEGRATION TESTS ==="

python3 "$TEST_DIR/integration/test_pipeline.py" "$PROJECT_ROOT" 2>/dev/null && {
    INT_RESULT=$(python3 "$TEST_DIR/integration/test_pipeline.py" "$PROJECT_ROOT" 2>&1 | tail -1)
    INT_PASSED=$(echo "$INT_RESULT" | grep -oP '\d+(?= passed)' || echo "0")
    INT_TOTAL=$(echo "$INT_RESULT" | grep -oP '\d+(?= total)' || echo "0")
    PASSED_TESTS=$((PASSED_TESTS + INT_PASSED))
    TOTAL_TESTS=$((TOTAL_TESTS + INT_TOTAL))
} || log_fail "Integration tests"

echo ""
echo "========================================"
echo "SPARK TEST SUMMARY"
echo "========================================"
echo "Total:  $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

# Write results
cat > "$RESULTS_DIR/spark_test_results.json" << EOF
{
  "version": "0.8.6",
  "timestamp": "$(date -Iseconds)",
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "pass_rate": "$(echo "scale=2; $PASSED_TESTS * 100 / ($TOTAL_TESTS + 1)" | bc 2>/dev/null || echo "N/A")%"
}
EOF

[ $FAILED_TESTS -eq 0 ] && exit 0 || exit 1
