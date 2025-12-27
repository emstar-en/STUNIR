#!/usr/bin/env bash
# scripts/lib/spec_to_ir_files.sh
# Shell stub for IR File Splitting.

stunir_shell_spec_to_ir_files() {
    local out_root=""
    local manifest_out=""
    local bundle_out=""
    local bundle_manifest_out=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-root) out_root="$2"; shift 2 ;;
            --manifest-out) manifest_out="$2"; shift 2 ;;
            --bundle-out) bundle_out="$2"; shift 2 ;;
            --bundle-manifest-out) bundle_manifest_out="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -z "$out_root" ]]; then
        echo "Error: --out-root required" >&2
        return 1
    fi

    mkdir -p "$out_root"
    mkdir -p "$(dirname "$manifest_out")"
    mkdir -p "$(dirname "$bundle_out")"

    echo "WARNING: Shell implementation of spec_to_ir_files is a STUB."
    echo "         It does not actually split IR into files."

    # Create dummy artifacts to satisfy build.sh
    echo "{}" > "$manifest_out"
    touch "$bundle_out"
    echo "{}" > "$bundle_manifest_out"
}
