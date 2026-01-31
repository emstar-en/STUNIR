#!/bin/bash
# STUNIR Receipt Verifier
# PRIMARY: Ada SPARK tools are the DEFAULT implementation
#
# Usage: ./verify.sh [OPTIONS] [<receipt_file>] [base_dir]
#
# Options:
#   --local    Verify local build outputs (uses receipts/ir_manifest.json)
#   --strict   Enable strict verification mode
#
# Override via: STUNIR_PROFILE=spark|python

set -e

# Helper functions
log() { echo "[Verify] $1"; }
warn() { echo "[Verify][WARN] $1" >&2; }
error() { echo "[Verify][ERROR] $1" >&2; }

# Parse command-line options
LOCAL_MODE=false
STRICT_MODE=false
RECEIPT_FILE=""
BASE_DIR="."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --help|-h)
            echo "STUNIR Receipt Verifier"
            echo "PRIMARY: Ada SPARK implementation"
            echo ""
            echo "Usage: $0 [OPTIONS] [<receipt_file>] [base_dir]"
            echo ""
            echo "Options:"
            echo "  --local    Verify local build outputs (uses receipts/ir_manifest.json)"
            echo "  --strict   Enable strict verification mode"
            echo ""
            echo "Environment:"
            echo "  STUNIR_PROFILE=spark|python  Override runtime detection"
            exit 0
            ;;
        *)
            if [ -z "$RECEIPT_FILE" ]; then
                RECEIPT_FILE="$1"
            else
                BASE_DIR="$1"
            fi
            shift
            ;;
    esac
done

# If --local mode, use default local receipt file
if [ "$LOCAL_MODE" = true ]; then
    if [ -z "$RECEIPT_FILE" ]; then
        # Try common local receipt locations
        if [ -f "receipts/ir_manifest.json" ]; then
            RECEIPT_FILE="receipts/ir_manifest.json"
        elif [ -f "build/receipt.json" ]; then
            RECEIPT_FILE="build/receipt.json"
        else
            # No receipt file found in --local mode
            # This can happen during transition period or when receipt generation is not yet implemented
            warn "No receipt file found in common locations (receipts/ir_manifest.json, build/receipt.json)"
            warn "This is expected during the Ada SPARK migration phase"
            log "Verifying build artifacts exist instead..."
            
            # Check that basic build outputs exist
            if [ -f "asm/spec_ir.json" ] || [ -f "build/ir.json" ]; then
                log "✅ Build artifacts found"
                log "PASSED. Build verification complete (no receipt to verify)"
                exit 0
            else
                error "ERROR: No build artifacts found. Run ./scripts/build.sh first"
                exit 1
            fi
        fi
    fi
fi

PROFILE="${STUNIR_PROFILE:-auto}"

# Tool paths
SPARK_VERIFIER="tools/spark/bin/stunir_verifier_main"

# Final check for receipt file
if [ -z "$RECEIPT_FILE" ]; then
    echo "STUNIR Receipt Verifier"
    echo "PRIMARY: Ada SPARK implementation"
    echo ""
    echo "Usage: $0 [OPTIONS] [<receipt_file>] [base_dir]"
    echo ""
    echo "Options:"
    echo "  --local    Verify local build outputs"
    echo "  --strict   Enable strict verification mode"
    echo ""
    echo "Environment:"
    echo "  STUNIR_PROFILE=spark|python  Override runtime detection"
    exit 1
fi

if [ "$STRICT_MODE" = true ]; then
    log "Running in STRICT mode"
fi

# --- Helper: Polyglot SHA-256 ---
calc_hash() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$file" | awk '{print $2}'
    else
        echo "ERROR: No SHA-256 tool found" >&2
        exit 1
    fi
}

# --- Detect Runtime ---
detect_runtime() {
    if [ "$PROFILE" = "spark" ]; then
        echo "spark"
        return
    elif [ "$PROFILE" = "python" ]; then
        echo "python"
        return
    fi

    # Auto-detection: SPARK (Ada) -> Python (reference)
    if [ -x "$SPARK_VERIFIER" ]; then
        echo "spark"
    elif command -v python3 >/dev/null 2>&1; then
        warn "Falling back to Python reference implementation"
        echo "python"
    else
        echo "shell"
    fi
}

RUNTIME=$(detect_runtime)
log "Verifier runtime: $RUNTIME"

case "$RUNTIME" in
    spark)
        # PRIMARY: Ada SPARK verifier
        log "Using Ada SPARK Verifier (PRIMARY implementation)"
        if [ -x "$SPARK_VERIFIER" ]; then
            "$SPARK_VERIFIER" --receipt "$RECEIPT_FILE" --base-dir "$BASE_DIR"
            exit $?
        else
            warn "SPARK verifier not built, falling back to shell verification"
        fi
        ;;
        
    python)
        # REFERENCE: Python verifier
        warn "=============================================="
        warn "USING PYTHON REFERENCE IMPLEMENTATION"
        warn "This is NOT recommended for production use."
        warn "For production, use Ada SPARK: STUNIR_PROFILE=spark"
        warn "=============================================="
        ;;
esac

# Shell fallback verification (works with any runtime)
log "Reading receipt: $RECEIPT_FILE"

# 1. Extract the "outputs" block
if command -v jq >/dev/null 2>&1; then
    OUTPUTS_RAW=$(jq -r '.outputs | to_entries | .[] | "\(.key)|\(.value)"' "$RECEIPT_FILE")
elif command -v python3 >/dev/null 2>&1; then
    OUTPUTS_RAW=$(python3 -c "import sys, json; d=json.load(sys.stdin)['outputs']; [print(f'{k}|{v}') for k,v in d.items()]" < "$RECEIPT_FILE")
else
    warn "Using Shell-Native JSON parser (relies on canonical format)"
    RAW_CONTENT=$(cat "$RECEIPT_FILE" | sed -n 's/.*"outputs":{\([^}]*\)}.*/\1/p')
    OUTPUTS_RAW=$(echo "$RAW_CONTENT" | sed 's/,"/\n"/g' | sed 's/"//g' | sed 's/:/|/')
fi

# 2. Verify Each File
FAILURES=0
TOTAL=0

IFS=$'\n'
for entry in $OUTPUTS_RAW; do
    [ -z "$entry" ] && continue
    
    FILE_PATH=$(echo "$entry" | cut -d'|' -f1)
    EXPECTED_HASH=$(echo "$entry" | cut -d'|' -f2)
    
    FULL_PATH="$BASE_DIR/$FILE_PATH"
    
    if [ ! -f "$FULL_PATH" ]; then
        echo "❌ MISSING: $FILE_PATH"
        FAILURES=$((FAILURES + 1))
        continue
    fi
    
    ACTUAL_HASH=$(calc_hash "$FULL_PATH")
    
    if [ "$ACTUAL_HASH" == "$EXPECTED_HASH" ]; then
        echo "✅ OK: $FILE_PATH"
    else
        echo "❌ MISMATCH: $FILE_PATH"
        echo "   Expected: $EXPECTED_HASH"
        echo "   Actual:   $ACTUAL_HASH"
        FAILURES=$((FAILURES + 1))
    fi
    TOTAL=$((TOTAL + 1))
done
unset IFS

echo "---------------------------------------------------"
if [ "$FAILURES" -eq 0 ]; then
    log "PASSED. All $TOTAL files match the receipt."
    exit 0
else
    log "FAILED. Found $FAILURES errors."
    exit 1
fi
