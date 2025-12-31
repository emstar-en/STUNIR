#!/bin/bash
# STUNIR Shell Library
# Provides common functions for Profile 3 (Shell-Native) execution

log_info() {
    echo "ℹ️  $1" >&2
}

log_success() {
    echo "✅ $1" >&2
}

log_error() {
    echo "❌ $1" >&2
}

# Check for required tools
require_tool() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Missing required tool: $1"
        exit 1
    fi
}

# Canonicalize JSON (Shell implementation using jq or python fallback)
canonicalize_json() {
    local input_file="$1"

    if command -v jq &> /dev/null; then
        # jq -c -S . "$input_file" # Compact, Sorted
        # Note: jq's compact output might differ slightly from RFC 8785 (e.g. float formatting)
        # For Profile 3, we assume simple types.
        jq -c -S . "$input_file"
    elif command -v python3 &> /dev/null; then
        python3 -c "import sys, json; print(json.dumps(json.load(open('$input_file')), sort_keys=True, separators=(',', ':'), ensure_ascii=False))"
    else
        log_error "No JSON processor found (jq or python3 required for canonicalization)"
        exit 1
    fi
}
