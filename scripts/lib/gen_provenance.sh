#!/usr/bin/env bash
# scripts/lib/gen_provenance.sh

stunir_shell_gen_provenance() {
    local out_json=""
    local out_header=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-json) out_json="$2"; shift 2 ;;
            --out-header) out_header="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -n "$out_json" ]]; then
        mkdir -p "$(dirname "$out_json")"
        echo '{"provenance": "shell_mock"}' > "$out_json"
    fi

    if [[ -n "$out_header" ]]; then
        mkdir -p "$(dirname "$out_header")"
        echo '#define PROVENANCE_MOCK 1' > "$out_header"
    fi

    echo "Generated Shell Provenance"
}
