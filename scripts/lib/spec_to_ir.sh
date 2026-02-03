#!/usr/bin/env bash
# scripts/lib/spec_to_ir.sh

stunir_shell_spec_to_ir() {
    local out=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out) out="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -n "$out" ]]; then
        mkdir -p "$(dirname "$out")"
        echo "SHELL_IR_MANIFEST_MOCK" > "$out"
        echo "Generated Shell IR Summary at $out"
    fi
}
