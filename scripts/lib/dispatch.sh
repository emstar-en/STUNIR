#!/bin/bash
# STUNIR Dispatch Library
# FIX V3: Forces shell implementation for receipts to bypass native issues.

# Explicitly override any auto-detection for receipt generation
STUNIR_USE_NATIVE_RECEIPT=0
STUNIR_USE_SHELL_RECEIPT=1

log_info "Dispatch: Forcing Shell+Inline-Python receipt generation (Fix V3)"

# Source the receipt library
source "$(dirname "${BASH_SOURCE[0]}")/receipt.sh"
