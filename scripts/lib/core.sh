#!/bin/sh
# STUNIR Shell-Native Core Utilities
# Provides logging, error handling, and hashing abstractions.

set -u

# --- Logging ---
log_info() { echo "[INFO] $1" >&2; }
log_warn() { echo "[WARN] $1" >&2; }
log_err()  { echo "[ERROR] $1" >&2; }

fail() {
    log_err "$1"
    exit 1
}

# --- Hashing ---
# Detect available sha256 tool
if command -v sha256sum >/dev/null 2>&1; then
    HASHER="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASHER="shasum -a 256"
else
    # Fallback for very minimal systems (unlikely to work for verification but allows bootstrapping)
    log_warn "No sha256sum or shasum found. Hashing disabled."
    HASHER="echo '0000000000000000000000000000000000000000  -'"
fi

calc_sha256() {
    # Usage: calc_sha256 "filename"
    if [ -f "$1" ]; then
        $HASHER "$1" | awk '{print $1}'
    else
        echo "0000000000000000000000000000000000000000"
    fi
}

calc_sha256_string() {
    # Usage: echo "string" | calc_sha256_string
    $HASHER | awk '{print $1}'
}

# --- Environment ---
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1" || fail "Could not create directory: $1"
    fi
}
