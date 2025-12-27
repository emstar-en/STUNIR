#!/usr/bin/env bash
# scripts/lib/json_canon.sh
# Shell-native JSON Canonicalization (Profile 3)
# Limitations:
# - Does not handle nested objects/arrays recursively (requires recursion/stack).
# - Assumes flat or simple depth.
# - Used primarily for generating manifests/receipts.

stunir_json_simple_object() {
    local -a keys=()
    local -A map

    while [[ $# -gt 0 ]]; do
        local k="$1"
        local v="$2"
        keys+=("$k")
        map["$k"]="$v"
        shift 2
    done

    # Sort keys
    IFS=$'\n' sorted_keys=($(sort <<<"${keys[*]}"))
    unset IFS

    echo -n "{"
    local first=1
    for k in "${sorted_keys[@]}"; do
        if [[ -z "$k" ]]; then continue; fi

        if [[ "$first" == "0" ]]; then echo -n ","; fi
        first=0

        local v="${map[$k]}"
        # Escape backslashes first
        v="${v//\\/\\\\}"
        # Escape quotes
        v="${v//\"/\\\"}"
        # Escape newlines (basic)
        v="${v//$'
'/\\n}"

        echo -n "\"\$k\":\"$v\""
    done
    echo "}"
}
