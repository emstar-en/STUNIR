#!/usr/bin/env bash
set -euo pipefail

# STUNIR Toolchain Discovery
# Generates a lockfile of the current environment's tools.

OUTPUT_FILE="${1:-build/local_toolchain.lock.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Define critical tools to capture
TOOLS=("python3" "git" "bash" "sh" "make" "gcc" "cargo")

echo "{" > "$OUTPUT_FILE"
echo "  \"schema\": \"stunir_toolchain_v1\"," >> "$OUTPUT_FILE"
echo "  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"," >> "$OUTPUT_FILE"
echo "  \"tools\": {" >> "$OUTPUT_FILE"

FIRST=true
for tool in "${TOOLS[@]}"; do
    if command -v "$tool" >/dev/null 2>&1; then
        path=$(command -v "$tool")
        
        # Compute hash (platform agnostic)
        if command -v sha256sum >/dev/null 2>&1; then
            hash=$(sha256sum "$path" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            hash=$(shasum -a 256 "$path" | awk '{print $1}')
        else
            hash="unknown_no_hasher"
        fi

        # Version string (best effort)
        version=$("$tool" --version 2>&1 | head -n 1 | sed 's/"/\\"/g' || echo "unknown")

        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> "$OUTPUT_FILE"
        fi

        echo "    \"$tool\": {" >> "$OUTPUT_FILE"
        echo "      \"path\": \"$path\"," >> "$OUTPUT_FILE"
        echo "      \"sha256\": \"$hash\"," >> "$OUTPUT_FILE"
        echo "      \"version\": \"$version\"" >> "$OUTPUT_FILE"
        echo "    }" >> "$OUTPUT_FILE"
    fi
done

echo "" >> "$OUTPUT_FILE"
echo "  }" >> "$OUTPUT_FILE"
echo "}" >> "$OUTPUT_FILE"

echo "Toolchain lockfile generated at: $OUTPUT_FILE"
