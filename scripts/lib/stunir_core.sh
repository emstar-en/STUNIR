#!/usr/bin/env bash
# scripts/lib/stunir_core.sh
# Core library for STUNIR shell scripts

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[STUNIR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[STUNIR:WARN]${NC} $1"
}

log_err() {
    echo -e "${RED}[STUNIR:ERROR]${NC} $1" >&2
}

generate_manifest() {
    local target_dir="$1"
    local output_file="$2"

    if [ ! -d "$target_dir" ]; then
        log_warn "Manifest target directory '$target_dir' does not exist."
        return
    fi

    log_info "Generating manifest for $target_dir -> $output_file"

    # Simple manifest: relative path + sha256
    # Use find to list files, sort for determinism
    (cd "$target_dir" && find . -type f -not -path '*/.*' | sort | while read -r file; do
        # Remove leading ./
        clean_file="${file#./}"
        # Calculate hash
        if command -v sha256sum >/dev/null 2>&1; then
            hash=$(sha256sum "$file" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            hash=$(shasum -a 256 "$file" | awk '{print $1}')
        else
            hash="nohash"
        fi
        echo "$hash  $clean_file"
    done) > "$output_file"
}
