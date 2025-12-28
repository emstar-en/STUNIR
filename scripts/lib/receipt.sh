#!/usr/bin/env bash

stunir_shell_receipt() {
    local target=""
    local out_file=""
    local toolchain_lock=""
    local epoch="0"

    # Parse Arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --target)
                target="$2"
                shift 2
                ;;
            --out)
                out_file="$2"
                shift 2
                ;;
            --toolchain-lock)
                toolchain_lock="$2"
                shift 2
                ;;
            --epoch)
                epoch="$2"
                shift 2
                ;;
            *)
                # Handle legacy positional arguments if any
                if [[ -z "$target" && "$1" != --* ]]; then target="$1"; shift; continue; fi
                if [[ -z "$out_file" && "$1" != --* ]]; then out_file="$1"; shift; continue; fi
                if [[ -z "$toolchain_lock" && "$1" != --* ]]; then toolchain_lock="$1"; shift; continue; fi
                shift
                ;;
        esac
    done

    # Validation
    if [[ -z "$target" ]]; then echo "ERROR: Missing --target"; exit 1; fi
    if [[ -z "$out_file" ]]; then echo "ERROR: Missing --out"; exit 1; fi
    if [[ -z "$toolchain_lock" ]]; then echo "ERROR: Missing --toolchain-lock"; exit 1; fi

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
    if [[ -x "build/stunir_native" ]] && [[ "${STUNIR_USE_NATIVE_RECEIPT:-0}" == "1" ]]; then
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

    # Calculate ID on canonical core JSON (stunir-json-c14n-v1)
    local core_c14n
    core_c14n=$(echo "$json_content" | jq -cS .)

    local core_id
    core_id=$(printf '%s' "$core_c14n" | sha256sum | awk '{print $1}')

    # Add ID and write canonical receipt JSON
    echo "$core_c14n" | jq --arg id "$core_id" '.receipt_core_id_sha256 = $id' | jq -cS . > "$out_file"
}
