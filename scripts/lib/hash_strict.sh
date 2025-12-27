#!/usr/bin/env bash

# STUNIR Strict Hashing Library
# Implements deterministic hashing for files and directories (Manifest Mode).

stunir_compute_strict_hash() {
    local target="$1"

    if [[ -f "$target" ]]; then
        # Single File: Simple Content Hash
        sha256sum "$target" | awk '{print $1}'
<<<<<<< HEAD
        
    elif [[ -d "$target" ]]; then
        # Directory: Strict Manifest Hash
        # 1. Enter directory (forces relative paths)
        # 2. Find all files (print0 handles spaces/newlines)
        # 3. Sort deterministically (LC_ALL=C)
        # 4. Compute hash for each file
        # 5. Hash the resulting manifest list
=======

    elif [[ -d "$target" ]]; then
        # Directory: Strict Manifest Hash
>>>>>>> origin/rescue/main-pre-force
        (
            cd "$target" && \
            find . -type f -print0 | \
            LC_ALL=C sort -z | \
            xargs -0 sha256sum | \
            sha256sum | \
            awk '{print $1}'
        )
    else
        echo "ERROR: Target not found: $target" >&2
        return 1
    fi
}
