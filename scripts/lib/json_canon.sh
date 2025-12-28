#!/usr/bin/env bash

stunir_canon_echo() {
    # Usage: stunir_canon_echo '{"key": "value"}' > output.json
    local json_content="$1"

    if command -v python3 >/dev/null 2>&1; then
        # Python is the gold standard for STUNIR canonicalization
        # FIXED: ensure_ascii=False to match verifier
        python3 -c "import json, sys; print(json.dumps(json.loads(sys.argv[1]), separators=(',', ':'), sort_keys=True, ensure_ascii=False))" "$json_content"
    elif command -v jq >/dev/null 2>&1; then
        # jq -c -S (compact, sorted) matches ensure_ascii=False (raw UTF-8)
        echo "$json_content" | jq -c -S .
    else
        # Fallback: Strip whitespace manually (DANGEROUS but better than nothing for simple objects)
        echo "$json_content" | tr -d '[:space:]'
    fi
}
