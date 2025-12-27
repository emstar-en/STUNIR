#!/usr/bin/env bash

stunir_shell_compile_provenance() {
    local in_prov=""
    local out_bin=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --in-prov) in_prov="$2"; shift 2 ;;
            --out-bin) out_bin="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    echo "Compiling Provenance (Shell Mode: Copy)..."
    cp "$in_prov" "$out_bin"
}
