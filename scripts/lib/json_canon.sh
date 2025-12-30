#!/usr/bin/env bash
# scripts/lib/json_canon.sh
# Shell-Native JSON Canonicalization Helpers
#
# STRATEGY:
# We do not parse arbitrary JSON. We construct JSON deterministically.
# 1. Keys must be sorted.
# 2. No whitespace between separators.
# 3. Strings must be properly escaped.

stunir_json_escape() {
    local input="$1"
    # Escape backslash, double quote, newline, tab
    # This is a minimal escaper.
    echo -n "$input" | sed 's/\\/\\\\/g; s/"/\\"/g; s/\t/\\t/g' | tr '\n' ' '
}

# Usage: stunir_canon_object "key1" "val1" "key2" "val2" ...
# keys MUST be passed in sorted order if you want canonical output!
# This function does NOT sort for you (shell sorting is tricky with arrays).
# It assumes the CALLER provides sorted keys.
stunir_canon_object() {
    echo -n "{"
    local first=1
    while [[ $# -gt 0 ]]; do
        local key="$1"
        local val="$2"
        shift 2

        if [[ $first -eq 0 ]]; then
            echo -n ","
        fi
        echo -n "\"$key\":\"$val\""
        first=0
    done
    echo -n "}"
}

# Usage: stunir_canon_echo '{"a":1}'
# Just echoes the argument. In shell mode, we often cheat and just echo the pre-calculated string.
stunir_canon_echo() {
    echo "$1"
}
