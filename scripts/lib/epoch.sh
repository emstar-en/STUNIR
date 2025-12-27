#!/usr/bin/env bash
# scripts/lib/epoch.sh

stunir_shell_epoch() {
    local out_json=""
    local set_epoch=""
    local print_epoch=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-json) out_json="$2"; shift 2 ;;
            --set-epoch) set_epoch="$2"; shift 2 ;;
            --print-epoch) print_epoch=1; shift ;;
            *) shift ;;
        esac
    done

    local epoch="${set_epoch:-0}"

    if [[ "$print_epoch" == "1" ]]; then
        echo "$epoch"
    fi

    if [[ -n "$out_json" ]]; then
        mkdir -p "$(dirname "$out_json")"
        echo "{"selected_epoch": "$epoch"}" > "$out_json"
    fi
}
