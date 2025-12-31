#!/usr/bin/env bash
# STUNIR Receipt Generation (Shell-Native)

source "$STUNIR_ROOT/scripts/lib/json.sh"

stunir_generate_receipt() {
    local build_dir="$STUNIR_ROOT/build"
    local lockfile="$build_dir/local_toolchain.lock.json"
    local receipt_file="$build_dir/receipt.json"

    stunir_log "Generating Receipt..."

    # 1. Hash Inputs
    # For now, we assume a default spec if none provided, or just hash the lockfile
    local lock_hash=$(stunir_hash_file "$lockfile")

    # 2. Hash Outputs
    # Scan build dir for outputs (excluding receipt and lockfile)
    # This is a simplified "Manifest" approach

    json_init
    json_start_object
    json_key_val "schema_version" "1.0.0"
    json_add_comma
    json_key_val "timestamp" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    json_add_comma

    # Toolchain Section
    json_key_start_array "toolchain"
    json_start_object
    json_key_val "type" "lockfile"
    json_add_comma
    json_key_val "path" "build/local_toolchain.lock.json"
    json_add_comma
    json_key_val "sha256" "$lock_hash"
    json_end_object
    json_end_array
    json_add_comma

    # Outputs Section
    json_key_start_array "outputs"

    local first=true
    for f in "$build_dir"/*; do
        local fname=$(basename "$f")
        if [[ "$fname" == "receipt.json" || "$fname" == "local_toolchain.lock.json" ]]; then
            continue
        fi

        if [ "$first" = true ]; then
            first=false
        else
            json_add_comma
        fi

        local fhash=$(stunir_hash_file "$f")
        json_start_object
        json_key_val "path" "build/$fname"
        json_add_comma
        json_key_val "sha256" "$fhash"
        json_end_object
    done

    json_end_array

    json_end_object

    echo "$JSON_BUFFER" > "$receipt_file"
    stunir_ok "Receipt generated at: $receipt_file"
}
