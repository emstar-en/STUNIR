#!/usr/bin/env bash

stunir_generate_toolchain() {
    local out_file="$1"
    mkdir -p "$(dirname "$out_file")"

    # 1. Define required tools
    local tools=("python3" "git" "bash" "sh" "rm" "cp" "mkdir" "cat" "jq" "sha256sum")
    
    # 2. Build JSON object incrementally
    local json_content="{}"
    
    # Add metadata (hostname is NOT deterministic across machines, but stable on one machine)
    # Ideally we should remove hostname for pure reproducibility, but for now let's keep it stable locally.
    json_content=$(echo "$json_content" | jq --arg h "$(hostname)" '.hostname = $h')
    json_content=$(echo "$json_content" | jq --arg p "$OSTYPE" '.platform = $p')

    # Add tools
    local tools_json="{}"
    for tool in "${tools[@]}"; do
        local path
        path=$(command -v "$tool" || echo "")
        
        if [[ -n "$path" ]]; then
            # Normalize path (forward slashes)
            path=$(echo "$path" | sed 's/\\/\//g')
            
            # Calculate hash (if file exists)
            local hash="unknown"
            if [[ -f "$path" ]]; then
                hash=$(sha256sum "$path" | awk '{print $1}')
            fi
            
            # Add to tools object
            tools_json=$(echo "$tools_json" | jq --arg t "$tool" --arg p "$path" --arg h "$hash" \
                '.[$t] = {path: $p, sha256: $h}')
        else
            tools_json=$(echo "$tools_json" | jq --arg t "$tool" '.[$t] = {status: "missing"}')
        fi
    done

    # Merge tools into main JSON
    json_content=$(echo "$json_content" | jq --argjson t "$tools_json" '.tools = $t')

    # 3. WRITE CANONICAL JSON (Sorted Keys)
    echo "$json_content" | jq --sort-keys . > "$out_file"
    
    echo "Toolchain lockfile generated at: $out_file"
}
