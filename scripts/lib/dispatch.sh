#!/bin/bash
set -e

log_info() { echo "[INFO] $1"; }

stunir_dispatch() {
    local issue_id="$1"
    log_info "Dispatching $issue_id"

    case "$issue_id" in
        ISSUE.HASKELL.*)    scripts/build_haskell_first.sh ;;
        ISSUE.IR.*)         scripts/canonicalize.sh ;;
        ISSUE.BUILD.*)      scripts/build.sh ;;
        ISSUE.NATIVE.*)     tools/native/haskell/bin/stunir-native-hs ;;
        ISSUE.PACK.*)       tools/pack_attestation/generate_root.sh ;;
        ISSUE.PROV.*)       scripts/lib/gen_provenance.sh ;;
        ISSUE.RECEIPT.*)    scripts/lib/receipt.sh ;;
        *) log_info "Unknown: $issue_id"; return 1 ;;
    esac
    log_info "$issue_id: COMPLETE"
}

"$@"