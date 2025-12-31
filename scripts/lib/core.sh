#!/usr/bin/env bash
# STUNIR Shell-Native Core Library

stunir_fail() {
    echo "[ERROR] $1" >&2
    exit 1
}

stunir_log() {
    echo "[INFO] $1"
}

# Simple JSON value extractor (grep/sed based, fragile but dependency-free)
# Usage: json_get_key "key" "file.json"
json_get_key() {
    local key=$1
    local file=$2
    grep -o "\"\$key\": *\"[^\"]*\"" "$file" | sed 's/.*: *"\(.*\)"/\1/'
}
