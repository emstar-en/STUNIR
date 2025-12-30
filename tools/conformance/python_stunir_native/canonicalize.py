#!/usr/bin/env python3
import sys
import json
import cbor2
import hashlib

def canonicalize(spec_path, ir_path):
    with open(spec_path) as f:
        spec = json.load(f)

    # Profile-3: no floats, canonical sort
    ir_cbor = cbor2.dumps(spec["spec"], canonical=True, float_policy="forbid")

    sha256 = hashlib.sha256(ir_cbor).hexdigest()
    print(f"IR: {sha256}")

    with open(ir_path, "wb") as f:
        f.write(ir_cbor)

if __name__ == "__main__":
    canonicalize(sys.argv[1], sys.argv[2])
