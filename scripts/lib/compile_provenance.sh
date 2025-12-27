#!/usr/bin/env bash
# scripts/lib/compile_provenance.sh
# Shell implementation for compiling the provenance emitter.

stunir_shell_compile_provenance() {
    local epoch=""
    local epoch_json=""
    local provenance_json=""
    local ir_manifest=""
    local bundle_manifest=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --epoch) epoch="$2"; shift 2 ;;
            --epoch-json) epoch_json="$2"; shift 2 ;;
            --provenance-json) provenance_json="$2"; shift 2 ;;
            --ir-manifest) ir_manifest="$2"; shift 2 ;;
            --bundle-manifest) bundle_manifest="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    local src="tools/prov_emit.c"
    local bin="bin/prov_emit"
    local receipt="receipts/prov_emit.json"

    mkdir -p bin receipts

    # Check for CC (exported by toolchain.sh)
    if [[ -z "${CC:-}" ]]; then
        echo "WARNING: No C compiler found (CC not set). Skipping prov_emit."
        # Record skipped receipt
        stunir_dispatch record_receipt             --target "$bin"             --receipt "$receipt"             --status "SKIPPED_NO_CC"             --build-epoch "$epoch"             --epoch-json "$epoch_json"             --inputs "$provenance_json"
        return 0
    fi

    echo "Compiling $bin with $CC..."

    # Compile
    # We use the CC variable directly.
    # Note: In strict mode, CC should be an absolute path from toolchain.
    "$CC" -std=c11 -O2 -Wno-builtin-macro-redefined         -D_FORTIFY_SOURCE=2         -D_STUNIR_BUILD_EPOCH="$epoch"         -Ibuild         -o "$bin" "$src"

    local ret=$?

    if [[ $ret -eq 0 ]]; then
        # Record success receipt
        stunir_dispatch record_receipt             --target "$bin"             --receipt "$receipt"             --status "COMPILED"             --build-epoch "$epoch"             --epoch-json "$epoch_json"             --inputs "$provenance_json" "$ir_manifest" "$bundle_manifest" "$src"
    else
        echo "ERROR: Compilation failed."
        # Record failure receipt
        stunir_dispatch record_receipt             --target "$bin"             --receipt "$receipt"             --status "FAILED_COMPILATION"             --build-epoch "$epoch"             --epoch-json "$epoch_json"
        return $ret
    fi
}
