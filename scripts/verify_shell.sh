#!/usr/bin/env bash
# scripts/verify_shell.sh
# STUNIR Shell-Native Verifier
# Dependencies: grep, sed, shasum (or sha256sum)
# NO Python, NO jq, NO compiled binaries.

set -e
export LC_ALL=C

log_info() { echo "[STUNIR:SHELL-VERIFY] $1"; }
log_err() { echo "[STUNIR:SHELL-VERIFY] ERROR: $1" >&2; }

IR_FILE="$1"
PROV_FILE="$2"

if [ -z "$IR_FILE" ] || [ -z "$PROV_FILE" ]; then
    echo "Usage: $0 <ir_file> <prov_file>"
    exit 1
fi

if [ ! -f "$IR_FILE" ]; then
    log_err "IR file not found: $IR_FILE"
    exit 1
fi

if [ ! -f "$PROV_FILE" ]; then
    log_err "Provenance file not found: $PROV_FILE"
    exit 1
fi

# 1. Calculate Hash of IR
# Detect hashing tool
if command -v sha256sum >/dev/null 2>&1; then
    HASHER="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASHER="shasum -a 256"
else
    log_err "No SHA256 tool found (need sha256sum or shasum)"
    exit 1
fi

log_info "Hashing IR file..."
# Calculate hash and grab the first field (the hex string)
CALC_HASH=$($HASHER "$IR_FILE" | awk '{print $1}')
log_info "Calculated Hash: $CALC_HASH"

# 2. Extract Hash from Provenance (Pure Shell JSON parsing is fragile, but we use regex for this specific field)
# Looking for: "ir_hash": "sha256:<HEX>"
# We use sed to extract the value inside the quotes after ir_hash
log_info "Reading Provenance..."

# Grep the line, then sed to extract. 
# Assumption: "ir_hash" is on its own line or we can match the pattern reliably.
# Pattern: "ir_hash"\s*:\s*"sha256:([a-f0-9]+)"
EXTRACTED_HASH=$(grep -o '"ir_hash"[[:space:]]*:[[:space:]]*"sha256:[a-f0-9]*"' "$PROV_FILE" | sed -E 's/.*"sha256:([a-f0-9]+)".*//')

if [ -z "$EXTRACTED_HASH" ]; then
    log_err "Could not extract ir_hash from provenance file."
    exit 1
fi

log_info "Expected Hash:   $EXTRACTED_HASH"

# 3. Compare
if [ "$CALC_HASH" == "$EXTRACTED_HASH" ]; then
    echo "✅ VERIFICATION SUCCESS: Artifacts match."
    exit 0
else
    echo "❌ VERIFICATION FAILED: Hash mismatch."
    exit 1
fi
