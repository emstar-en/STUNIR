#!/bin/bash
# STUNIR Receipt Verifier
# Usage: ./verify.sh <receipt_file> [base_dir]

set -e

RECEIPT_FILE="$1"
BASE_DIR="${2:-.}"

if [ -z "$RECEIPT_FILE" ]; then
    echo "Usage: $0 <receipt_file> [base_dir]"
    exit 1
fi

# --- Helper: Polyglot SHA-256 (Same as receipt.sh) ---
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

echo "[Verify] Reading receipt: $RECEIPT_FILE"

# 1. Extract the "outputs" block
# We use a polyglot strategy to get the JSON content of "outputs"
if command -v jq >/dev/null 2>&1; then
    # Easy mode
    OUTPUTS_RAW=$(jq -r '.outputs | to_entries | .[] | "\(.key)|\(.value)"' "$RECEIPT_FILE")
elif command -v python3 >/dev/null 2>&1; then
    # Python mode
    OUTPUTS_RAW=$(python3 -c "import sys, json; d=json.load(sys.stdin)['outputs']; [print(f'{k}|{v}') for k,v in d.items()]" < "$RECEIPT_FILE")
else
    # Shell mode (The "Desperate" Parser)
    # Relies on Canonical JSON format: {"key":"value","key2":"value2"}
    # 1. Extract everything between "outputs":{ and the closing }
    # 2. Split by comma
    # 3. Clean up quotes
    echo "[Verify] Warning: Using Shell-Native JSON parser (relies on canonical format)"
    
    # Extract the outputs object content. 
    # This sed is specific to the canonical format produced by json_canon.sh
    RAW_CONTENT=$(cat "$RECEIPT_FILE" | sed -n 's/.*"outputs":{\([^}]*\)}.*/\1/p')
    
    # Split "file":"hash","file2":"hash2" into lines
    OUTPUTS_RAW=$(echo "$RAW_CONTENT" | sed 's/,"/\n"/g' | sed 's/"//g' | sed 's/:/|/')
fi

# 2. Verify Each File
FAILURES=0
TOTAL=0

# Set Internal Field Separator to newline to handle the loop correctly
IFS=$'\n'
for entry in $OUTPUTS_RAW; do
    [ -z "$entry" ] && continue
    
    # Split "filename|hash"
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
    echo "[Verify] PASSED. All $TOTAL files match the receipt."
    exit 0
else
    echo "[Verify] FAILED. Found $FAILURES errors."
    exit 1
fi
