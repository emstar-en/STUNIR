#!/bin/sh
# STUNIR Shell-Native Runner
# Orchestrates the build process in Shell mode.

set -u

BASE_DIR=$(dirname "$0")
. "$BASE_DIR/core.sh"
. "$BASE_DIR/manifest.sh"
. "$BASE_DIR/receipt.sh"

run_shell_build() {
    log_info "Starting STUNIR Shell-Native Build..."

    # 1. Discovery
    generate_lockfile

    # 2. (Stub) IR Generation
    # In a full implementation, this would process spec/*.json
    log_info "IR Generation: [SKIPPED] (Shell stub)"

    # 3. Receipt
    generate_receipt

    log_info "Build Complete."
}

# Execute if run directly
if [ "$(basename "$0")" = "runner.sh" ]; then
    run_shell_build
fi
