#!/usr/bin/env bash

# Renamed to match dispatch.sh expectation
stunir_shell_receipt() {
    local target="$1"
    local out_file="$2"
    local toolchain_lock="$3"
    local epoch="${4:-0}"

    # Ensure target exists
    if [[ ! -f "$target" ]]; then
        echo "ERROR: Target file not found for receipt: $target"
        exit 1
    fi

    # Ensure toolchain lock exists
    if [[ ! -f "$toolchain_lock" ]]; then
        echo "ERROR: Toolchain lock not found: $toolchain_lock"
        exit 1
    fi

    # NATIVE PATH: Use Rust binary if available
    if [[ -x "build/stunir_native" ]]; then
        ./build/stunir_native gen-receipt \
            --target "$target" \
            --toolchain "$toolchain_lock" \
            --epoch "$epoch" \
            --out "$out_file"
        return
    fi

    # FALLBACK: Shell Implementation (jq + openssl)
    local target_hash
    target_hash=$(sha256sum "$target" | awk '{print $1}')
    
    local toolchain_hash
    toolchain_hash=$(sha256sum "$toolchain_lock" | awk '{print $1}')

    # Construct JSON
    local json_content
    json_content=$(jq -n \
        --arg schema "stunir.receipt.build.v1" \
        --arg status "success" \
        --argjson epoch "$epoch" \
        --arg target "$target" \
        --arg sha256 "$target_hash" \
        --arg toolchain_sha256 "$toolchain_hash" \
        '{
            schema: $schema,
            status: $status,
            build_epoch: $epoch,
            epoch: {selected_epoch: $epoch, source: "default"},
            target: $target,
            sha256: $sha256,
            toolchain_sha256: $toolchain_sha256,
            argv: ["shell_generated"],
            inputs: [],
            tool: null
        }')

    # Calculate ID
    local core_id
    core_id=$(echo "$json_content" | sha256sum | awk '{print $1}')

    # Add ID and Write
    echo "$json_content" | jq --arg id "$core_id" '.receipt_core_id_sha256 = $id' > "$out_file"
}
