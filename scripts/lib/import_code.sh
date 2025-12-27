#!/usr/bin/env bash

stunir_shell_import_code() {
    local input_root=""
    local out_spec=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --input-root) input_root="$2"; shift 2 ;;
            --out-spec) out_spec="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Importing Code from $input_root (Shell Mode)..."
    
    # Start JSON
    echo '{ "kind": "spec", "modules": [' > "$out_spec"
    
    # Iterate files (Simple implementation: assumes no crazy filenames)
    first=1
    for f in "$input_root"/*; do
        if [[ -f "$f" ]]; then
            if [[ "$first" == "0" ]]; then echo "," >> "$out_spec"; fi
            
            name=$(basename "$f")
            # Read content and escape newlines/quotes (Basic Shell Escaping)
            # Note: This is fragile for complex code, but works for this test.
            content=$(cat "$f" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | awk '{printf "%s\\n", $0}')
            
            echo "    { \"name\": \"$name\", \"code\": \"$content\" }" >> "$out_spec"
            first=0
        fi
    done
    
    # End JSON
    echo ']}' >> "$out_spec"
}
