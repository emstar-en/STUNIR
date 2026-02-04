#!/usr/bin/env python3
"""
STUNIR Epoch Tool - Timestamp Generation Utility

Generates canonical JSON output containing the current Unix timestamp
and ISO 8601 formatted time. Used for build reproducibility and
versioning in the STUNIR pipeline.

Output Format (Canonical JSON - RFC 8785 / JCS subset):
    {"iso_8601":"YYYY-MM-DDTHH:MM:SSZ","source":"stunir_epoch_tool","unix_timestamp":<int>}

Features:
    - Deterministic output (sorted keys, no whitespace)
    - UTC timestamps only
    - No external dependencies

Usage:
    python tools/epoch.py > timestamp.json
    python tools/epoch.py | jq .

Integration:
    Called by build scripts to embed build timestamps in generated artifacts.
    The canonical format ensures reproducible builds when the same timestamp
    is used across different tools and languages.
"""

import json
import time
import sys
import os


def main() -> None:
    """Generate and output canonical JSON timestamp."""
    # Capture standard epoch data
    now = time.time()
    data = {
        "unix_timestamp": int(now),
        "iso_8601": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "source": "stunir_epoch_tool"
    }

    # FIX V3: Output Strict Canonical JSON
    # No indentation, no spaces after separators, sorted keys.
    json.dump(data, sys.stdout, separators=(',', ':'), sort_keys=True)

    # Ensure a single newline at EOF is NOT added by print/dump default if we want pure blob,
    # but standard POSIX text files usually have one. 
    # For strict canonicalization (JCS), usually no trailing newline is preferred 
    # if it's treated as a blob, but for file storage, a newline is safer.
    # However, the user requested 'strict canonical JSON', which usually implies the raw string.
    # We will write to stdout without an extra newline from print.
    sys.stdout.flush()


if __name__ == "__main__":
    main()
