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

# Simple JSON value extractor (grep/sed based)
json_get_key() {
    local key=$1
    local file=$2
    grep -o "\"\$key\": *\"[^\"]*\"" "$file" | sed 's/.*: *"\(.*\)"/\1/'
}
