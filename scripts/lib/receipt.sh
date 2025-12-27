#!/usr/bin/env bash
# scripts/lib/receipt.sh

stunir_shell_record_receipt() {
    local target=""
    local receipt=""
    local status=""
    local epoch=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --target) target="$2"; shift 2 ;;
            --receipt) receipt="$2"; shift 2 ;;
            --status) status="$2"; shift 2 ;;
            --build-epoch) epoch="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -n "$receipt" ]]; then
        mkdir -p "$(dirname "$receipt")"
        echo "{"target": "$target", "status": "$status", "epoch": "$epoch"}" > "$receipt"
        echo "Recorded Shell Receipt: $receipt"
    fi
}
