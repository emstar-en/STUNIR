#!/bin/bash
IN_IR="$1"
OUT_FILE="$2"

if [ -z "$IN_IR" ] || [ -z "$OUT_FILE" ]; then
    echo "Usage: $0 <in_ir> <out_file>"
    exit 1
fi

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required for shell emission."
    exit 127
fi

jq -r -f scripts/lib/emit_bash.jq "$IN_IR" > "$OUT_FILE"
chmod +x "$OUT_FILE"
