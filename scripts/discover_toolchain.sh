#!/usr/bin/env bash
# scripts/discover_toolchain.sh
# Scans the host environment for required tools and generates a deterministic lockfile.
# Output: build/local_toolchain.lock.json

set -euo pipefail

# Default output
LOCKFILE="build/local_toolchain.lock.json"
mkdir -p build

# 1. Define the "Must Have" tools
# Format: logical_name|binary_name|optional_flag (0=required, 1=optional)
TOOLS=(
    "python|python3|0"
    "git|git|1"
    "bash|bash|0"
    "cc|cc|1"
    "rustc|rustc|1"
    "cargo|cargo|1"
    "stunir_native|stunir-native|1"
)

echo "{" > "$LOCKFILE"
echo "  \"tools\": {" >> "$LOCKFILE"

FIRST=1

for item in "${TOOLS[@]}"; do
    IFS='|' read -r LOGICAL BINARY OPTIONAL <<< "$item"

    # Resolve path (using 'command -v' which respects current PATH)
    PATH_RESULT="$(command -v "$BINARY" || true)"

    if [[ -z "$PATH_RESULT" ]]; then
        if [[ "$OPTIONAL" == "0" ]]; then
            echo "ERROR: Required tool '$BINARY' not found in PATH." >&2
            exit 1
        else
            echo "  Skipping optional tool: $BINARY" >&2
            continue
        fi
    fi

    # Normalize path (resolve symlinks if possible, though 'realpath' isn't always available)
    # Fallback to simple path if realpath is missing
    if command -v realpath >/dev/null 2>&1; then
        ABS_PATH="$(realpath "$PATH_RESULT")"
    else
        ABS_PATH="$PATH_RESULT"
    fi

    # Calculate SHA256 (portable-ish)
    if command -v sha256sum >/dev/null 2>&1; then
        HASH="$(sha256sum "$ABS_PATH" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        HASH="$(shasum -a 256 "$ABS_PATH" | awk '{print $1}')"
    else
        HASH="unknown_no_sha256_tool"
    fi

    # Get Version (best effort)
    VERSION_STR=""
    if [[ "$BINARY" == "python3" ]]; then
        VERSION_STR="$("$ABS_PATH" --version 2>&1 | head -n1)"
    elif [[ "$BINARY" == "git" ]]; then
        VERSION_STR="$("$ABS_PATH" --version 2>&1 | head -n1)"
    elif [[ "$BINARY" == "bash" ]]; then
        VERSION_STR="$("$ABS_PATH" --version 2>&1 | head -n1)"
    elif [[ "$BINARY" == "rustc" ]]; then
        VERSION_STR="$("$ABS_PATH" --version 2>&1 | head -n1)"
    fi
    # Escape quotes in version string
    VERSION_STR="${VERSION_STR//\"/\\\"}"

    # JSON Emission
    if [[ "$FIRST" == "0" ]]; then echo "," >> "$LOCKFILE"; fi
    FIRST=0

    echo "    \"$LOGICAL\": {" >> "$LOCKFILE"
    echo "      \"path\": \"$ABS_PATH\"," >> "$LOCKFILE"
    echo "      \"sha256\": \"$HASH\"," >> "$LOCKFILE"
    echo "      \"version\": \"$VERSION_STR\"" >> "$LOCKFILE"
    echo -n "    }" >> "$LOCKFILE"

    echo "  Locked $LOGICAL -> $ABS_PATH ($HASH)" >&2
done

echo "" >> "$LOCKFILE"
echo "  }," >> "$LOCKFILE"
echo "  \"host_platform\": \"$(uname -s)_$(uname -m)\"" >> "$LOCKFILE"
echo "}" >> "$LOCKFILE"

echo "Toolchain lockfile generated at $LOCKFILE"
