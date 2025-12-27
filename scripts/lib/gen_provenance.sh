#!/usr/bin/env bash

stunir_shell_gen_provenance() {
    local in_ir=""
    local out_prov=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --in-ir) in_ir="$2"; shift 2 ;;
            --out-prov) out_prov="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Generating Provenance (Shell Mode)..."
    cp "$in_ir" "$out_prov"
}
