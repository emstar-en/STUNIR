#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=${LC_ALL:-C}
export LANG=${LANG:-C}
export TZ=${TZ:-UTC}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

python3 -B tools/verify_build.py --repo . --strict
