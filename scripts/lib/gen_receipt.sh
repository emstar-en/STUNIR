#!/bin/bash

# Shell implementation of 'gen-receipt'
# Adheres to stunir.receipt.build.v1

stunir_gen_receipt() {
    local target="$1"
    local status="$2"
    local epoch="$3"
    local t_name="$4"
    local t_path="$5"
    local t_hash="$6"
    local t_ver="$7"
    shift 7
    local argv=("$@")

    # Construct JSON using jq if available, or manual string concat
    # For STUNIR shell-native, we try to be minimal.

    # Note: This is a simplified generator. 
    # Proper canonicalization is hard in shell.

    cat <<EOF
{
  "receipt_schema": "stunir.receipt.build.v1",
  "receipt_target": "$target",
  "receipt_status": "$status",
  "receipt_build_epoch": $epoch,
  "receipt_epoch_json": "build/epoch.json",
  "receipt_inputs": {},
  "receipt_tool": {
    "name": "$t_name",
    "path": "$t_path",
    "sha256": "$t_hash",
    "version": "$t_ver"
  },
  "receipt_argv": [$(printf '"%s",' "${argv[@]}" | sed 's/,$//')]
}
EOF
}
