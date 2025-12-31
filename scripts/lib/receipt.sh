#!/bin/sh
# STUNIR Receipt Generator
# Generates a basic receipt.json for the build

. "$(dirname "$0")/core.sh"
. "$(dirname "$0")/json.sh"

generate_receipt() {
    ensure_dir "receipts"
    RECEIPT_FILE="receipts/build_receipt.json"

    log_info "Generating receipt..."

    # Calculate input digest (simple recursive hash of spec/)
    if [ -d "spec" ]; then
        SPEC_DIGEST=$(find spec -type f -exec sha256sum {} + | sort | sha256sum | awk '{print $1}')
    else
        SPEC_DIGEST="empty"
    fi

    json_init "$RECEIPT_FILE"
    json_obj_start

    json_key_str "type" "stunir_receipt"
    json_key_str "profile" "shell_native"

    # Timestamp (allowed in receipt metadata)
    DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    json_key_str "timestamp" "$DATE"

    json_key_str "spec_digest" "$SPEC_DIGEST"

    # Toolchain Lock Hash
    if [ -f "local_toolchain.lock.json" ]; then
        LOCK_HASH=$(calc_sha256 "local_toolchain.lock.json")
        json_key_str "toolchain_lock_sha256" "$LOCK_HASH"
    fi

    json_key_str "status" "SUCCESS"

    json_obj_end

    log_info "Receipt written to $RECEIPT_FILE"
}
