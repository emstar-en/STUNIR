#!/usr/bin/env bash
# scripts/lib/spec_to_ir_files.sh

stunir_shell_spec_to_ir_files() {
    local out_root=""
    local manifest_out=""
    local bundle_manifest_out=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out-root) out_root="$2"; shift 2 ;;
            --manifest-out) manifest_out="$2"; shift 2 ;;
            --bundle-manifest-out) bundle_manifest_out="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    if [[ -n "$out_root" ]]; then
        mkdir -p "$out_root"
        echo "mock_ir_content" > "$out_root/dummy.dcbor"
    fi

    if [[ -n "$manifest_out" ]]; then
        mkdir -p "$(dirname "$manifest_out")"
        echo '{"files": []}' > "$manifest_out"
    fi

    if [[ -n "$bundle_manifest_out" ]]; then
        mkdir -p "$(dirname "$bundle_manifest_out")"
        echo '{"bundle": "mock"}' > "$bundle_manifest_out"
    fi

    echo "Generated Shell IR Files"
}
