#!/bin/bash
# STUNIR Rust Test Runner v0.8.6
# Runs all Rust pipeline tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUST_TOOLS="$PROJECT_ROOT/tools/rust"
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

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        log_pass "$test_name"
        return 0
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log_fail "$test_name"
        return 1
    fi
}

echo "========================================"
echo "STUNIR Rust Test Suite v0.8.6"
echo "========================================"
echo ""

# Build Rust tools
log_info "Building Rust tools..."
cd "$RUST_TOOLS"
cargo build --release 2>/dev/null || cargo build 2>/dev/null

SPEC_TO_IR="$RUST_TOOLS/target/release/stunir_spec_to_ir"
IR_TO_CODE="$RUST_TOOLS/target/release/stunir_ir_to_code"
[ -f "$SPEC_TO_IR" ] || SPEC_TO_IR="$RUST_TOOLS/target/debug/stunir_spec_to_ir"
[ -f "$IR_TO_CODE" ] || IR_TO_CODE="$RUST_TOOLS/target/debug/stunir_ir_to_code"

echo ""
echo "=== UNIT TESTS: spec_to_ir ==="

# Run Python test runner for comprehensive spec_to_ir tests
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
echo "RUST TEST SUMMARY"
echo "========================================"
echo "Total:  $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

# Write results
cat > "$RESULTS_DIR/rust_test_results.json" << EOF
{
  "version": "0.8.6",
  "timestamp": "$(date -Iseconds)",
  "total_tests": $TOTAL_TESTS,
  "passed": $PASSED_TESTS,
  "failed": $FAILED_TESTS,
  "pass_rate": "$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%"
}
EOF

[ $FAILED_TESTS -eq 0 ] && exit 0 || exit 1
