#!/usr/bin/env bash
set -euo pipefail

ENVELOPE=${1:-receipt.dsse.json}
shift || true

python3 tools/verify_build.py   --envelope "${ENVELOPE}"   --repo .   --tmp-dir _verify_build   "$@"
