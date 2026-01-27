#!/bin/bash
# STUNIR Strict Verification Script
# Issue: ISSUE.MANIFEST.0001 - Strict verify manifest fix
#
# Usage: ./verify_strict.sh [--local] [--strict] [--ir-only]
#
# Options:
#   --local    Run local verification (no network)
#   --strict   Enable strict mode (IR manifest must match exactly)
#   --ir-only  Only verify IR manifest (skip other receipts)

set -e

SCRIPT_DIR="$(dirname "$0")"
BASE_DIR="${SCRIPT_DIR}/.."

# Parse arguments
LOCAL_MODE=false
STRICT_MODE=false
IR_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --local)  LOCAL_MODE=true ;;
        --strict) STRICT_MODE=true ;;
        --ir-only) IR_ONLY=true ;;
        --help)
            echo "STUNIR Strict Verification"
            echo "Usage: $0 [--local] [--strict] [--ir-only]"
            exit 0
            ;;
    esac
done

log() { echo "[verify_strict] $1"; }

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

# --- IR Manifest Verification ---
verify_ir_manifest() {
    local manifest_path="$BASE_DIR/receipts/ir_manifest.json"
    local ir_dir="$BASE_DIR/asm/ir"
    local failures=0
    local verified=0
    
    log "Verifying IR manifest: $manifest_path"
    
    # Check manifest exists
    if [ ! -f "$manifest_path" ]; then
        if [ "$STRICT_MODE" = true ]; then
            log "FAIL: IR manifest not found (strict mode)"
            return 1
        else
            log "WARN: IR manifest not found, skipping"
            return 0
        fi
    fi
    
    # Check IR directory exists
    if [ ! -d "$ir_dir" ]; then
        if [ "$STRICT_MODE" = true ]; then
            log "FAIL: IR directory not found: $ir_dir"
            return 1
        else
            log "WARN: IR directory not found, skipping"
            return 0
        fi
    fi
    
    # Parse manifest using Python (most portable for JSON)
    if command -v python3 >/dev/null 2>&1; then
        # Get list of files from manifest
        manifest_files=$(python3 -c "
import json
import sys
with open('$manifest_path', 'r') as f:
    manifest = json.load(f)
for entry in manifest.get('files', []):
    print(f\"{entry['filename']}|{entry['sha256']}\")
" 2>/dev/null)
        
        if [ -z "$manifest_files" ]; then
            log "WARN: No files in manifest"
            if [ "$STRICT_MODE" = true ]; then
                # In strict mode, check if IR dir has any dcbor files
                dcbor_count=$(find "$ir_dir" -name "*.dcbor" 2>/dev/null | wc -l)
                if [ "$dcbor_count" -gt 0 ]; then
                    log "FAIL: Manifest empty but $dcbor_count dcbor files exist"
                    return 1
                fi
            fi
            return 0
        fi
        
        # Verify each file in manifest
        echo "$manifest_files" | while IFS='|' read -r filename expected_hash; do
            [ -z "$filename" ] && continue
            
            file_path="$ir_dir/$filename"
            
            if [ ! -f "$file_path" ]; then
                log "❌ MISSING: $filename"
                echo "FAIL"
                continue
            fi
            
            actual_hash=$(calc_hash "$file_path")
            
            if [ "$actual_hash" = "$expected_hash" ]; then
                log "✅ OK: $filename"
            else
                log "❌ MISMATCH: $filename"
                log "   Expected: $expected_hash"
                log "   Actual:   $actual_hash"
                echo "FAIL"
            fi
        done | grep -c "FAIL" || true
        
        failures=$(echo "$manifest_files" | while IFS='|' read -r filename expected_hash; do
            [ -z "$filename" ] && continue
            file_path="$ir_dir/$filename"
            if [ ! -f "$file_path" ]; then
                echo "FAIL"
            else
                actual_hash=$(calc_hash "$file_path")
                if [ "$actual_hash" != "$expected_hash" ]; then
                    echo "FAIL"
                fi
            fi
        done | grep -c "FAIL" || echo "0")
        
        # In strict mode, verify no extra files exist
        if [ "$STRICT_MODE" = true ]; then
            log "Checking for extra files (strict mode)..."
            
            # Get files from disk
            disk_files=$(find "$ir_dir" -name "*.dcbor" -exec basename {} \; 2>/dev/null | sort)
            
            # Get files from manifest
            manifest_file_list=$(python3 -c "
import json
with open('$manifest_path', 'r') as f:
    manifest = json.load(f)
for entry in sorted(manifest.get('files', []), key=lambda x: x['filename']):
    print(entry['filename'])
" 2>/dev/null)
            
            # Compare
            extra_files=$(comm -23 <(echo "$disk_files" | sort) <(echo "$manifest_file_list" | sort) 2>/dev/null || true)
            
            if [ -n "$extra_files" ]; then
                log "❌ EXTRA FILES not in manifest:"
                echo "$extra_files" | while read -r f; do
                    log "   - $f"
                done
                failures=$((failures + $(echo "$extra_files" | wc -l)))
            fi
        fi
        
    elif command -v jq >/dev/null 2>&1; then
        # jq fallback
        jq -r '.files[] | "\(.filename)|\(.sha256)"' "$manifest_path" | while IFS='|' read -r filename expected_hash; do
            file_path="$ir_dir/$filename"
            if [ -f "$file_path" ]; then
                actual_hash=$(calc_hash "$file_path")
                if [ "$actual_hash" = "$expected_hash" ]; then
                    log "✅ OK: $filename"
                else
                    log "❌ MISMATCH: $filename"
                    failures=$((failures + 1))
                fi
            else
                log "❌ MISSING: $filename"
                failures=$((failures + 1))
            fi
        done
    else
        log "WARN: No JSON parser available (need python3 or jq)"
        return 0
    fi
    
    if [ "$failures" -gt 0 ]; then
        log "IR manifest verification FAILED with $failures errors"
        return 1
    else
        log "IR manifest verification PASSED"
        return 0
    fi
}

# --- Main ---
log "STUNIR Strict Verification"
log "Mode: local=$LOCAL_MODE, strict=$STRICT_MODE"

TOTAL_FAILURES=0

# Verify IR manifest
if ! verify_ir_manifest; then
    TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
fi

# Exit if IR-only mode
if [ "$IR_ONLY" = true ]; then
    if [ "$TOTAL_FAILURES" -gt 0 ]; then
        log "VERIFICATION FAILED"
        exit 1
    else
        log "VERIFICATION PASSED"
        exit 0
    fi
fi

# Additional receipt verification would go here
# For now, delegate to standard verify.sh if receipt exists
if [ -f "$BASE_DIR/receipts/build_receipt.json" ] && [ "$IR_ONLY" = false ]; then
    log "Verifying build receipt..."
    "$SCRIPT_DIR/verify.sh" "$BASE_DIR/receipts/build_receipt.json" "$BASE_DIR" || TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
fi

# Summary
log "---------------------------------------------------"
if [ "$TOTAL_FAILURES" -gt 0 ]; then
    log "VERIFICATION FAILED with $TOTAL_FAILURES error(s)"
    exit 1
else
    log "VERIFICATION PASSED"
    exit 0
fi
