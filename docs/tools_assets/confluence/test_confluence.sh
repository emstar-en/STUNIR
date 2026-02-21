#!/usr/bin/env bash
#
# STUNIR Confluence Test Suite
#
# This script verifies that all four pipelines (SPARK, Python, Rust, Haskell)
# produce bitwise-identical outputs from the same inputs.
#
# Usage: ./test_confluence.sh [--verbose] [--category CATEGORY]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_VECTORS_DIR="$SCRIPT_DIR/test_vectors"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VERBOSE=0
CATEGORY_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=1
            shift
            ;;
        --category)
            CATEGORY_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log() {
    echo -e "${BLUE}[CONFLUENCE]${NC} $*"
}

log_success() {
    echo -e "${GREEN}✅${NC} $*"
}

log_failure() {
    echo -e "${RED}❌${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}⚠️${NC} $*"
}

# Detect available pipelines
detect_pipelines() {
    local available=()
    
    # SPARK
    if [[ -x "$ROOT_DIR/tools/spark/bin/stunir_spec_to_ir_main" ]]; then
        available+=("SPARK")
    fi
    
    # Python
    if [[ -f "$ROOT_DIR/tools/spec_to_ir.py" ]]; then
        available+=("Python")
    fi
    
    # Rust
    if [[ -f "$ROOT_DIR/tools/rust/Cargo.toml" ]]; then
        available+=("Rust")
    fi
    
    # Haskell
    if [[ -f "$ROOT_DIR/tools/haskell/stunir-tools.cabal" ]]; then
        available+=("Haskell")
    fi
    
    echo "${available[@]}"
}

# Test spec_to_ir for all pipelines
test_spec_to_ir() {
    local test_name="$1"
    local spec_file="$2"
    
    log "Testing spec_to_ir: $test_name"
    
    local -A outputs
    local -A hashes
    
    # SPARK
    if command -v "$ROOT_DIR/tools/spark/bin/stunir_spec_to_ir_main" &> /dev/null; then
        local output_file="$RESULTS_DIR/${test_name}_spark_ir.json"
        "$ROOT_DIR/tools/spark/bin/stunir_spec_to_ir_main" "$spec_file" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        outputs[SPARK]="$output_file"
        hashes[SPARK]=$(sha256sum "$output_file" | awk '{print $1}')
    fi
    
    # Python
    if [[ -f "$ROOT_DIR/tools/spec_to_ir.py" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_python_ir.json"
        python3 "$ROOT_DIR/tools/spec_to_ir.py" "$spec_file" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        outputs[Python]="$output_file"
        hashes[Python]=$(sha256sum "$output_file" | awk '{print $1}')
    fi
    
    # Rust
    if [[ -f "$ROOT_DIR/tools/rust/Cargo.toml" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_rust_ir.json"
        cd "$ROOT_DIR/tools/rust"
        cargo run --quiet --bin stunir_spec_to_ir -- "$spec_file" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        cd "$ROOT_DIR"
        outputs[Rust]="$output_file"
        hashes[Rust]=$(sha256sum "$output_file" | awk '{print $1}')
    fi
    
    # Haskell
    if [[ -f "$ROOT_DIR/tools/haskell/stunir-tools.cabal" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_haskell_ir.json"
        cd "$ROOT_DIR/tools/haskell"
        cabal run stunir_spec_to_ir -- "$spec_file" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        cd "$ROOT_DIR"
        outputs[Haskell]="$output_file"
        hashes[Haskell]=$(sha256sum "$output_file" | awk '{print $1}')
    fi
    
    # Compare hashes
    local first_hash=""
    local all_match=true
    
    for pipeline in "${!hashes[@]}"; do
        if [[ -z "$first_hash" ]]; then
            first_hash="${hashes[$pipeline]}"
        elif [[ "${hashes[$pipeline]}" != "$first_hash" ]]; then
            all_match=false
            break
        fi
    done
    
    if [[ "$all_match" == "true" ]]; then
        log_success "spec_to_ir: $test_name - All ${#hashes[@]} pipelines produce identical IR"
        return 0
    else
        log_failure "spec_to_ir: $test_name - Output divergence detected"
        for pipeline in "${!hashes[@]}"; do
            echo "  $pipeline: ${hashes[$pipeline]}"
        done
        return 1
    fi
}

# Test ir_to_code for all pipelines
test_ir_to_code() {
    local test_name="$1"
    local ir_file="$2"
    local target="$3"
    
    log "Testing ir_to_code: $test_name -> $target"
    
    local -A outputs
    local -A hashes
    
    # SPARK
    if command -v "$ROOT_DIR/tools/spark/bin/stunir_ir_to_code_main" &> /dev/null; then
        local output_file="$RESULTS_DIR/${test_name}_${target}_spark.txt"
        "$ROOT_DIR/tools/spark/bin/stunir_ir_to_code_main" "$ir_file" --target="$target" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        if [[ -f "$output_file" ]]; then
            outputs[SPARK]="$output_file"
            hashes[SPARK]=$(sha256sum "$output_file" | awk '{print $1}')
        fi
    fi
    
    # Python
    if [[ -f "$ROOT_DIR/tools/ir_to_code.py" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_${target}_python.txt"
        python3 "$ROOT_DIR/tools/ir_to_code.py" "$ir_file" --target="$target" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        if [[ -f "$output_file" ]]; then
            outputs[Python]="$output_file"
            hashes[Python]=$(sha256sum "$output_file" | awk '{print $1}')
        fi
    fi
    
    # Rust
    if [[ -f "$ROOT_DIR/tools/rust/Cargo.toml" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_${target}_rust.txt"
        cd "$ROOT_DIR/tools/rust"
        cargo run --quiet --bin stunir_ir_to_code -- "$ir_file" -t "$target" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        cd "$ROOT_DIR"
        if [[ -f "$output_file" ]]; then
            outputs[Rust]="$output_file"
            hashes[Rust]=$(sha256sum "$output_file" | awk '{print $1}')
        fi
    fi
    
    # Haskell
    if [[ -f "$ROOT_DIR/tools/haskell/stunir-tools.cabal" ]]; then
        local output_file="$RESULTS_DIR/${test_name}_${target}_haskell.txt"
        cd "$ROOT_DIR/tools/haskell"
        cabal run stunir_ir_to_code -- "$ir_file" -t "$target" -o "$output_file" 2>&1 | grep -v "^\[STUNIR\]" || true
        cd "$ROOT_DIR"
        if [[ -f "$output_file" ]]; then
            outputs[Haskell]="$output_file"
            hashes[Haskell]=$(sha256sum "$output_file" | awk '{print $1}')
        fi
    fi
    
    # Compare hashes
    if [[ ${#hashes[@]} -eq 0 ]]; then
        log_warning "ir_to_code: $test_name -> $target - No implementations available"
        return 2
    fi
    
    local first_hash=""
    local all_match=true
    
    for pipeline in "${!hashes[@]}"; do
        if [[ -z "$first_hash" ]]; then
            first_hash="${hashes[$pipeline]}"
        elif [[ "${hashes[$pipeline]}" != "$first_hash" ]]; then
            all_match=false
            break
        fi
    done
    
    if [[ "$all_match" == "true" ]]; then
        log_success "ir_to_code: $test_name -> $target - All ${#hashes[@]} pipelines produce identical code"
        return 0
    else
        log_failure "ir_to_code: $test_name -> $target - Output divergence detected"
        for pipeline in "${!hashes[@]}"; do
            echo "  $pipeline: ${hashes[$pipeline]}"
        done
        return 1
    fi
}

# Main test execution
main() {
    log "STUNIR Confluence Test Suite"
    log "Date: $(date +%Y-%m-%d)"
    log ""
    
    local available_pipelines
    available_pipelines=$(detect_pipelines)
    log "Available pipelines: $available_pipelines"
    log ""
    
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local skipped_tests=0
    
    # Test 1: Minimal spec
    if [[ -f "$TEST_VECTORS_DIR/minimal.json" ]]; then
        total_tests=$((total_tests + 1))
        if test_spec_to_ir "minimal" "$TEST_VECTORS_DIR/minimal.json"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
    fi
    
    # Test 2: Simple spec
    if [[ -f "$TEST_VECTORS_DIR/simple.json" ]]; then
        total_tests=$((total_tests + 1))
        if test_spec_to_ir "simple" "$TEST_VECTORS_DIR/simple.json"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests=$((failed_tests + 1))
        fi
    fi
    
    # Generate summary report
    log ""
    log "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
    log "Confluence Test Summary"
    log "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "="
    log "Total Tests: $total_tests"
    log_success "Passed: $passed_tests"
    log_failure "Failed: $failed_tests"
    log_warning "Skipped: $skipped_tests"
    
    local confluence_score=0
    if [[ $total_tests -gt 0 ]]; then
        confluence_score=$((passed_tests * 100 / total_tests))
    fi
    
    log ""
    log "Confluence Score: $confluence_score%"
    
    if [[ $confluence_score -eq 100 ]]; then
        log_success "🎉 Perfect confluence achieved!"
        exit 0
    elif [[ $confluence_score -ge 90 ]]; then
        log_warning "⚠️  Near confluence ($confluence_score%) - minor issues remain"
        exit 1
    else
        log_failure "❌ Confluence not achieved ($confluence_score%) - significant divergence"
        exit 1
    fi
}

main "$@"
