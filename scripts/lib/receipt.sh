#!/bin/bash
# STUNIR Native Receipt Generator
# Usage: ./receipt.sh --spec <spec_file> --lock <lock_file> --out-dir <dir> --dest <receipt_file>

set -e

# --- Configuration ---
SPEC_FILE=""
LOCK_FILE=""
OUT_DIR=""
DEST_FILE=""

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --spec) SPEC_FILE="$2"; shift ;;
        --lock) LOCK_FILE="$2"; shift ;;
        --out-dir) OUT_DIR="$2"; shift ;;
        --dest) DEST_FILE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$SPEC_FILE" || -z "$LOCK_FILE" || -z "$OUT_DIR" || -z "$DEST_FILE" ]]; then
    echo "Usage: $0 --spec <file> --lock <file> --out-dir <dir> --dest <file>"
    exit 1
fi

# --- Helper: Polyglot SHA-256 ---
calc_hash() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$file" | awk '{print $2}'
    else
        echo "ERROR: No SHA-256 tool found (sha256sum, shasum, openssl)" >&2
        exit 1
    fi
}

echo "[Receipt] Generating proof for build..."

# 1. Hash Inputs (Canonicalized)
# We must canonicalize the spec first to get the "Semantic Hash"
# (We use a temp file for the canonical spec)
TEMP_CANON_SPEC=$(mktemp)
./scripts/lib/json_canon.sh "$SPEC_FILE" > "$TEMP_CANON_SPEC"
SPEC_HASH=$(calc_hash "$TEMP_CANON_SPEC")
rm "$TEMP_CANON_SPEC"

# Hash the lockfile (assumed already canonical from discovery)
LOCK_HASH=$(calc_hash "$LOCK_FILE")

# 2. Hash Outputs
# We build a JSON string for the "outputs" map
OUTPUTS_JSON=""
# Find all files in OUT_DIR, sort them for determinism
# We use 'find' but need to handle paths carefully
while IFS= read -r file; do
    # Skip directories
    [ -d "$file" ] && continue
    
    # Calculate Hash
    FILE_HASH=$(calc_hash "$file")
    
    # Get relative path (strip OUT_DIR)
    # We use python or sed to robustly get relative path if possible, 
    # but for shell-native we'll do simple string replacement
    REL_PATH=${file#$OUT_DIR/}
    # Remove leading slash if present
    REL_PATH=${REL_PATH#/}
    
    # Append to JSON string (comma handling is tricky in loop, we'll fix trailing comma later)
    OUTPUTS_JSON="$OUTPUTS_JSON \"$REL_PATH\": \"$FILE_HASH\","
    
    echo "  - Hashed: $REL_PATH"
done < <(find "$OUT_DIR" -type f | sort)

# Remove trailing comma
OUTPUTS_JSON=${OUTPUTS_JSON%,}

# 3. Assemble Raw Receipt JSON
# We construct a "messy" JSON string, then let json_canon.sh fix it.
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

RAW_JSON="{
  \"schema\": \"stunir.receipt.v1\",
  \"meta\": {
    \"generator\": \"stunir-shell-native\",
    \"timestamp\": \"$TIMESTAMP\"
  },
  \"inputs\": {
    \"spec_sha256\": \"$SPEC_HASH\",
    \"toolchain_sha256\": \"$LOCK_HASH\"
  },
  \"outputs\": { $OUTPUTS_JSON }
}"

# 4. Canonicalize and Save
echo "$RAW_JSON" > "$DEST_FILE.tmp"
./scripts/lib/json_canon.sh "$DEST_FILE.tmp" > "$DEST_FILE"
rm "$DEST_FILE.tmp"

echo "[Receipt] Success! Saved to $DEST_FILE"
