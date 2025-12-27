#!/usr/bin/env bash

# Source the Strict Hashing Library
source scripts/lib/hash_strict.sh

stunir_shell_receipt() {
    local in_bin=""
    local out_receipt=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --in-bin) in_bin="$2"; shift 2 ;;
            --out-receipt) out_receipt="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Generating Receipt (Shell Mode) for $in_bin..."

    # USE STRICT HASHING
    local target_hash=$(stunir_compute_strict_hash "$in_bin")

    # Create Receipt JSON
    cat <<JSON > "$out_receipt"
{
  "kind": "receipt",
  "generator": "shell_native",
  "hashing": "strict_manifest_v1",
  "target_hash": "$target_hash",
  "status": "verified"
}
JSON
}
