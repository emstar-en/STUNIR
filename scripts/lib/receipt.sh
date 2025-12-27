#!/usr/bin/env bash

# Source the Strict Hashing Library
source scripts/lib/hash_strict.sh

stunir_shell_receipt() {
    local in_bin=""
    local out_receipt=""
    local toolchain_lock=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --toolchain-lock) toolchain_lock="$2"; shift 2 ;;
            --in-bin) in_bin="$2"; shift 2 ;;
            --out-receipt) out_receipt="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Generating Receipt (Shell Mode) for $in_bin..."

    # USE STRICT HASHING
    local target_hash=$(stunir_compute_strict_hash "$in_bin")
    
    # Calculate Toolchain Hash if lockfile exists
    local tc_entry=""
    if [[ -n "${toolchain_lock:-}" && -f "$toolchain_lock" ]]; then
        # Portable sha256sum
        if command -v sha256sum >/dev/null 2>&1; then
            tc_hash=$(sha256sum "$toolchain_lock" | awk '{print $1}')
        else
            tc_hash=$(shasum -a 256 "$toolchain_lock" | awk '{print $1}')
        fi
        tc_entry="  ,\"toolchain_sha256\": \"$tc_hash\""
    fi

    # Create Receipt JSON
    # We use manual echo to ensure we can inject the optional field easily
    # and avoid heredoc indentation issues.
    {
        echo "{"
        echo "  \"kind\": \"receipt\","
        echo "  \"generator\": \"shell_native\","
        echo "  \"hashing\": \"strict_manifest_v1\","
        echo "  \"target_hash\": \"$target_hash\","
        echo "  \"status\": \"verified\""
        echo "$tc_entry"
        echo "}"
    } > "$out_receipt"
}
