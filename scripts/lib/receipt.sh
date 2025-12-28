#!/bin/bash
# STUNIR Receipt Library
# Implements strict canonical JSON generation using inline Python 3.
# Guarantees byte-for-byte compliance with the verifier.

generate_receipt() {
    local output_file="$1"
    shift

    # We expect the remaining arguments to be key=value pairs
    # or we pull from specific environment variables.
    # This implementation passes the current environment to the inline python
    # to filter and format the receipt.

    python3 -c "
import sys
import json
import os

# Define the schema/keys we care about for the receipt
# This list would typically be aligned with the STUNIR spec
target_keys = [
    'STUNIR_PROFILE',
    'STUNIR_VERSION',
    'STUNIR_MODE',
    'REPO_URL',
    'COMMIT_SHA',
    'BUILD_TIMESTAMP',
    'HOST_PLATFORM'
]

receipt_data = {}

# 1. Ingest from Environment
for key in target_keys:
    if key in os.environ:
        receipt_data[key] = os.environ[key]

# 2. Ingest from Args (key=value)
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        receipt_data[k] = v

# 3. Strict Canonical JSON Generation
# - separators=(',', ':'): Removes whitespace
# - sort_keys=True: Deterministic ordering
# - ensure_ascii=False: Allow UTF-8 (though usually receipts are ASCII)
try:
    canonical_json = json.dumps(
        receipt_data, 
        separators=(',', ':'), 
        sort_keys=True, 
        ensure_ascii=False
    )
    print(canonical_json)
except Exception as e:
    sys.stderr.write(f'Error generating canonical receipt: {e}\n')
    sys.exit(1)
" "$@" > "$output_file"

    if [ $? -eq 0 ]; then
        echo "Receipt generated at $output_file"
    else
        echo "Failed to generate receipt"
        return 1
    fi
}
