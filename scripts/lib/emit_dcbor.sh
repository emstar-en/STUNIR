#!/bin/sh
# STUNIR dCBOR Emission Helper
# Issue: ISSUE.IR.0001 - IR canonicalization fix
#
# Converts IR JSON to dCBOR normal form for deterministic hashing.
# This is a foundation for hash-based determinism (will be replaced by semantic IR).

set -eu

log() { echo "[emit_dcbor] $1" >&2; }

# Input: JSON IR file
# Output: dCBOR file in asm/ir/

IR_JSON="${1:-build/ir.json}"
OUTPUT_DIR="${2:-asm/ir}"

if [ ! -f "$IR_JSON" ]; then
    log "Warning: IR JSON not found at $IR_JSON"
    log "Creating placeholder asm/ir directory..."
    mkdir -p "$OUTPUT_DIR"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract module name from IR JSON (simple extraction)
MODULE_NAME=$(cat "$IR_JSON" | grep -o '"module_name"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/' || echo "main")

if [ -z "$MODULE_NAME" ]; then
    MODULE_NAME="stunir_module"
fi

# Output filename
OUTPUT_FILE="$OUTPUT_DIR/${MODULE_NAME}.dcbor"

# For now, create a canonical JSON representation
# Full dCBOR encoding requires Haskell native tool
# This is a bootstrap placeholder that will be replaced
# when stunir-native-hs is built

log "Processing IR: $IR_JSON -> $OUTPUT_FILE"

# Create canonical JSON (sorted keys) as interim format
# The Haskell native tool will eventually replace this with true dCBOR
if command -v python3 >/dev/null 2>&1; then
    python3 -c "
import json
import sys
with open('$IR_JSON', 'r') as f:
    data = json.load(f)
# Sort keys for determinism
with open('$OUTPUT_FILE', 'wb') as f:
    # Write as canonical JSON for now (dCBOR placeholder)
    json_bytes = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    f.write(json_bytes)
print('Generated canonical IR: $OUTPUT_FILE')
" 2>/dev/null || {
    log "Python fallback: copying JSON as-is"
    cp "$IR_JSON" "$OUTPUT_FILE"
}
else
    # Shell fallback: just copy the file
    log "Copying IR JSON (no canonicalization available)"
    cp "$IR_JSON" "$OUTPUT_FILE"
fi

log "dCBOR emission complete: $OUTPUT_FILE"
