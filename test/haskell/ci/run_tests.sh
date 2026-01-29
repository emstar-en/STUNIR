#!/bin/bash
# STUNIR Conformance Tests - CI/CD Integration Script
# 
# This script is designed for CI/CD pipelines (GitHub Actions, GitLab CI, etc.)
# It handles setup, building, testing, and reporting.
#
# Usage:
#   ./ci/run_tests.sh [--setup] [--coverage] [--report]

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

DO_SETUP=0
DO_COVERAGE=0
DO_REPORT=0

for arg in "$@"; do
    case $arg in
        --setup)
            DO_SETUP=1
            ;;
        --coverage)
            DO_COVERAGE=1
            ;;
        --report)
            DO_REPORT=1
            ;;
        --help)
            echo "Usage: $0 [--setup] [--coverage] [--report]"
            echo ""
            echo "Options:"
            echo "  --setup     Install Haskell toolchain if missing"
            echo "  --coverage  Generate coverage report"
            echo "  --report    Generate JUnit XML report"
            exit 0
            ;;
    esac
done

cd "$PROJECT_DIR"

# Setup Haskell if requested
if [ $DO_SETUP -eq 1 ]; then
    log_info "Setting up Haskell toolchain..."
    if ! command -v ghc &> /dev/null; then
        log_info "Installing GHCup..."
        curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
        source ~/.ghcup/env
    fi
    log_info "GHC version: $(ghc --version)"
fi

# Check for build tools
if command -v stack &> /dev/null; then
    BUILD_TOOL="stack"
    log_info "Using Stack for building"
elif command -v cabal &> /dev/null; then
    BUILD_TOOL="cabal"
    log_info "Using Cabal for building"
else
    log_error "Neither stack nor cabal found. Install Haskell toolchain."
    exit 1
fi

# Install dependencies
log_info "Installing dependencies..."
if [ "$BUILD_TOOL" = "stack" ]; then
    stack setup --no-terminal
    stack build --only-dependencies --no-terminal
else
    cabal update
    cabal build --only-dependencies
fi

# Build
log_info "Building project..."
if [ "$BUILD_TOOL" = "stack" ]; then
    stack build --no-terminal
else
    cabal build all
fi

# Setup test data symlinks
log_info "Setting up test data..."
mkdir -p "$PROJECT_DIR/test_data"
ln -sf ../../../test_vectors/contracts "$PROJECT_DIR/test_data/contracts" 2>/dev/null || true
ln -sf ../../../test_vectors/native "$PROJECT_DIR/test_data/native" 2>/dev/null || true
ln -sf ../../../test_vectors/polyglot "$PROJECT_DIR/test_data/polyglot" 2>/dev/null || true
ln -sf ../../../test_vectors/receipts "$PROJECT_DIR/test_data/receipts" 2>/dev/null || true
ln -sf ../../../test_vectors/edge_cases "$PROJECT_DIR/test_data/edge_cases" 2>/dev/null || true
ln -sf ../../../test_vectors/property "$PROJECT_DIR/test_data/property" 2>/dev/null || true
ln -sf ../../../tests "$PROJECT_DIR/test_data/ir_bundle" 2>/dev/null || true

# Run tests
log_info "Running conformance tests (16 suites, 68+ test cases)..."
TEST_EXIT_CODE=0

if [ "$BUILD_TOOL" = "stack" ]; then
    if [ $DO_COVERAGE -eq 1 ]; then
        stack test --coverage --no-terminal || TEST_EXIT_CODE=$?
    else
        stack test --no-terminal || TEST_EXIT_CODE=$?
    fi
else
    if [ $DO_REPORT -eq 1 ]; then
        cabal test all --test-show-details=always || TEST_EXIT_CODE=$?
    else
        cabal test all || TEST_EXIT_CODE=$?
    fi
fi

# Generate report if requested
if [ $DO_REPORT -eq 1 ]; then
    log_info "Generating test report..."
    mkdir -p "$PROJECT_DIR/reports"
    # Generate simple report
    echo "STUNIR Conformance Test Report" > "$PROJECT_DIR/reports/test_report.txt"
    echo "Generated: $(date)" >> "$PROJECT_DIR/reports/test_report.txt"
    echo "Exit Code: $TEST_EXIT_CODE" >> "$PROJECT_DIR/reports/test_report.txt"
    log_info "Report saved to reports/test_report.txt"
fi

# Summary
echo ""
log_info "========================================"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_info "All tests PASSED"
else
    log_error "Some tests FAILED (exit code: $TEST_EXIT_CODE)"
fi
log_info "========================================"

exit $TEST_EXIT_CODE
