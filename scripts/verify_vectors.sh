#!/bin/bash
set -euo pipefail
source scripts/lib/stunir_lib.sh

echo "🧪 STUNIR Test Vector Verification"

# Ensure we have the production script
if [ ! -f "scripts/stunir_production.sh" ]; then
    log_error "scripts/stunir_production.sh not found. Please install STUNIR_PRODUCTION_CORRECTED.zip first."
    exit 1
fi

PASSED=0
FAILED=0

for spec in test_vectors/spec_*.json; do
    base=$(basename "$spec" .json)
    expected="test_vectors/${base/spec_/ir_}.expected.json"

    if [ ! -f "$expected" ]; then
        log_info "Skipping $spec (no expected output)"
        continue
    fi

    log_info "Testing $spec..."

    # Run spec-to-ir via the production dispatcher
    # We capture output to a temp file
    ACTUAL=$(scripts/stunir_production.sh spec-to-ir "$spec")

    # Compare
    EXPECTED_CONTENT=$(cat "$expected")

    if [ "$ACTUAL" == "$EXPECTED_CONTENT" ]; then
        log_success "$base matched"
        PASSED=$((PASSED + 1))
    else
        log_error "$base FAILED"
        echo "Expected: $EXPECTED_CONTENT"
        echo "Actual:   $ACTUAL"
        FAILED=$((FAILED + 1))
    fi
done

echo "----------------------------------------"
echo "Tests Passed: $PASSED"
echo "Tests Failed: $FAILED"

if [ "$FAILED" -eq 0 ]; then
    exit 0
else
    exit 1
fi
