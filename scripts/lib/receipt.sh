#!/usr/bin/env bash

stunir_shell_receipt() {
    local in_bin=""
    local out_receipt=""
    local toolchain_lock=""
    local epoch_json="build/epoch.json"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --toolchain-lock) toolchain_lock="$2"; shift 2 ;;
            --in-bin) in_bin="$2"; shift 2 ;;
            --out-receipt) out_receipt="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Generating Receipt for $in_bin..."
    
    # Use the Python tool for compliance
    if command -v python3 >/dev/null 2>&1; then
        python3 tools/gen_receipt.py \
            --target "$in_bin" \
            --out "$out_receipt" \
            --toolchain-lock "${toolchain_lock:-}" \
            --epoch-json "$epoch_json"
    else
        echo "ERROR: Python required for compliant receipt generation."
        exit 1
    fi
}
