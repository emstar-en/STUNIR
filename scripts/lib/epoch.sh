#!/usr/bin/env bash
source scripts/lib/json_canon.sh

stunir_shell_epoch() {
    local out_json=""
    local print_epoch=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-json) out_json="$2"; shift 2 ;;
            --print-epoch) print_epoch=true; shift ;;
            *) shift ;;
        esac
    done

    # Default to 0 for deterministic builds
    local epoch_val=0
    if [[ -n "${SOURCE_DATE_EPOCH:-}" ]]; then
        epoch_val="$SOURCE_DATE_EPOCH"
    fi

    if [[ "$print_epoch" == true ]]; then
        echo "$epoch_val"
    fi

    if [[ -n "$out_json" ]]; then
        # Generate Canonical JSON
        local json_body="{\"selected_epoch\": $epoch_val, \"source\": \"env\"}"
        stunir_canon_echo "$json_body" > "$out_json"
    fi
}
