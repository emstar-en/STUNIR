#!/usr/bin/env bash
# STUNIR Shell-Native Core Library

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

stunir_log() { echo -e "${BLUE}[INFO]${NC} $1"; }
stunir_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
stunir_err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
stunir_ok() { echo -e "${GREEN}[OK]${NC} $1"; }

stunir_fail() {
    stunir_err "$1"
    exit 1
}

# Calculate SHA256 of a file
# Usage: stunir_hash_file "filename"
stunir_hash_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # Empty hash
        return
    fi

    if command -v sha256sum >/dev/null; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        echo "unknown_no_hasher"
    fi
}

# Simple JSON value extractor (grep/sed based)
json_get_key() {
    local key=$1
    local file=$2
    grep -o "\"\$key\": *\"[^\"]*\"" "$file" | sed 's/.*: *"\(.*\)"/\1/'
}
