#!/usr/bin/env bash
# scripts/verify_shell.sh
# STUNIR Shell-Native Verifier
# Dependencies: grep, sed, shasum (or sha256sum), cut

set -e
export LC_ALL=C

# Source logging if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/lib/stunir_core.sh" ]; then
    source "$SCRIPT_DIR/lib/stunir_core.sh"
else
    log_info() { echo "[STUNIR:SHELL-VERIFY] $1"; }
    log_err() { echo "[STUNIR:SHELL-VERIFY] ERROR: $1" >&2; }
fi

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
if command -v sha256sum >/dev/null 2>&1; then
    HASHER="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASHER="shasum -a 256"
else
    log_err "No SHA256 tool found (need sha256sum or shasum)"
    exit 1
fi

log_info "Hashing IR file..."
CALC_HASH=$($HASHER "$IR_FILE" | awk '{print $1}')
log_info "Calculated Hash: $CALC_HASH"

# 2. Extract Hash from Provenance
log_info "Reading Provenance..."

# Find the line containing "ir_hash"
MATCHED_LINE=$(grep '"ir_hash"' "$PROV_FILE" | head -n 1)

if [ -z "$MATCHED_LINE" ]; then
    log_err "Could not find 'ir_hash' key in provenance file."
    log_err "File Content:"
    cat "$PROV_FILE"
    exit 1
fi

log_info "Found Provenance Line: $MATCHED_LINE"

# Robust extraction using cut
# Assumes standard JSON string format: "key": "value"
# Splitting by double-quote (") usually puts the value in field 4
# 1: indent/brace
# 2: key (ir_hash)
# 3: separator (: )
# 4: value (sha256:...)
RAW_VALUE=$(echo "$MATCHED_LINE" | cut -d'"' -f4)

# Remove "sha256:" prefix
EXTRACTED_HASH=${RAW_VALUE#sha256:}

if [ -z "$EXTRACTED_HASH" ]; then
    log_err "Could not extract hash value from line."
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
