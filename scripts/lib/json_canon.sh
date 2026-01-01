#!/bin/bash
# STUNIR JSON Canonicalizer (Polyglot)
# Usage: ./json_canon.sh <input_file>
# Output: Canonicalized JSON to stdout

set -e

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input_file>" >&2
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE" >&2
    exit 1
fi

# --- Strategy 1: STUNIR Native (The Enterprise Binary) ---
# Check for the binary in standard locations or PATH
if command -v stunir-native >/dev/null 2>&1; then
    stunir-native canon --file "$INPUT_FILE"
    exit 0
fi

# Check local build locations (Linux/Windows)
if [ -f "./stunir-native" ]; then
    ./stunir-native canon --file "$INPUT_FILE"
    exit 0
elif [ -f "./stunir-native.exe" ]; then
    ./stunir-native.exe canon --file "$INPUT_FILE"
    exit 0
fi

# --- Strategy 2: jq (The Shell Standard) ---
if command -v jq >/dev/null 2>&1; then
    # -c: Compact (no whitespace)
    # --sort-keys: Deterministic key ordering
    jq -c --sort-keys . "$INPUT_FILE"
    exit 0
fi

# --- Strategy 3: Python (The Forbidden Fruit) ---
# Only used if explicitly allowed or present
if command -v python3 >/dev/null 2>&1; then
    python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), sort_keys=True, separators=(',', ':')))" < "$INPUT_FILE"
    exit 0
fi

# --- Strategy 4: Perl (The Old Guard) ---
# Often available on 'minimal' Linux distros where Python is missing
if command -v perl >/dev/null 2>&1; then
    # Try to use core JSON::PP (Standard since Perl 5.14)
    if perl -MJSON::PP -e 1 >/dev/null 2>&1; then
        perl -MJSON::PP -e 'local $/; print JSON::PP->new->canonical->encode(decode_json(<>))' < "$INPUT_FILE"
        exit 0
    fi
fi

# --- Strategy 5: Pure Shell (The Desperate Fallback) ---
# WARNING: This CANNOT sort keys. It only removes whitespace.
# If we reach here, the Model MUST have outputted sorted keys.
echo "WARNING: Using pure shell fallback. Keys are NOT sorted. Determinism depends on input order." >&2

# Simple sed minifier:
# 1. Remove newlines
# 2. Remove spaces around structural characters
# Note: This is fragile with strings containing escaped quotes/spaces.
# For a robust enterprise silo, this is a 'Fail Open' risk.

# We use a safer 'tr' approach to just strip newlines and leading/trailing space
# This assumes the input is "Pretty Printed" standard JSON.
cat "$INPUT_FILE" | tr -d '\n' | sed 's/ : /:/g' | sed 's/, /,/g'
