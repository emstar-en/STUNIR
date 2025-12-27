#!/bin/bash
set -e
IN_SPEC="$1"
OUT_IR="$2"

if [ -z "$IN_SPEC" ] || [ -z "$OUT_IR" ]; then
    echo "Usage: $0 <in_spec> <out_ir>"
    exit 1
fi

jq -f scripts/lib/compile.jq "$IN_SPEC" > "$OUT_IR"
