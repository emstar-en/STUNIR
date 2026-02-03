#!/usr/bin/env sh
set -eu

python3 tools/verify_requirements.py --requirements receipts/requirements.json --deps_dir receipts/deps
