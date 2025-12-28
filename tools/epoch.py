#!/usr/bin/env python3
import json
import time
import sys
import os

def main():
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
