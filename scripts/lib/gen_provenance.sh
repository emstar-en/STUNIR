#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
EPOCH=$(scripts/lib/epoch.sh)
cat > build/provenance.json << EOF
{"epoch": $EPOCH, "roots": ["spec/", "asm/", "tools/"], "toolchain": $(scripts/discover_toolchain.sh | scripts/lib/json_canon.sh)}
EOF
echo "✓ build/provenance.json"