#!/usr/bin/env bash
# scripts/lib/json_canon.sh
# Minimal JSON emitter for STUNIR Shell Core.

# Usage: stunir_json_simple_object "key1" "val1" "key2" "val2" ...
# Emits a flat JSON object with sorted keys.
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

    # Sort keys (simple bubble sort or using sort command if available)
    # For shell portability, we'll rely on 'sort'
    IFS=$'\n' sorted_keys=($(sort <<<"${keys[*]}"))
    unset IFS

    echo -n "{"
    local first=1
    for k in "${sorted_keys[@]}"; do
        if [[ -z "$k" ]]; then continue; fi
        if [[ "$first" == "0" ]]; then echo -n ","; fi
        first=0

        local v="${map[$k]}"
        # Basic escaping (incomplete, but handles quotes)
        v="${v//\\/\\\\}"
        v="${v//\"/\\\"}"

        echo -n "\"$k\":\"$v\""
    done
    echo "}"
}
