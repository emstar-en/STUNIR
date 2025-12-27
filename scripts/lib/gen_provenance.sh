#!/usr/bin/env bash
<<<<<<< HEAD

stunir_shell_gen_provenance() {
    local in_ir=""
    local out_prov=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --in-ir) in_ir="$2"; shift 2 ;;
            --out-prov) out_prov="$2"; shift 2 ;;
=======
# scripts/lib/gen_provenance.sh

stunir_shell_gen_provenance() {
    local out_json=""
    local out_header=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-json) out_json="$2"; shift 2 ;;
            --out-header) out_header="$2"; shift 2 ;;
>>>>>>> origin/rescue/main-pre-force
            *) shift ;;
        esac
    done

<<<<<<< HEAD
    echo "Generating Provenance (Shell Mode)..."
    cp "$in_ir" "$out_prov"
=======
    if [[ -n "$out_json" ]]; then
        mkdir -p "$(dirname "$out_json")"
        echo '{"provenance": "shell_mock"}' > "$out_json"
    fi

    if [[ -n "$out_header" ]]; then
        mkdir -p "$(dirname "$out_header")"
        echo '#define PROVENANCE_MOCK 1' > "$out_header"
    fi

    echo "Generated Shell Provenance"
>>>>>>> origin/rescue/main-pre-force
}
