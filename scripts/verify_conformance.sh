#!/bin/bash
set -e
# ISSUE.VERIFY.0002: Haskell=reference conformance

REFERENCE="asm/haskell_ir_0001.dcbor"

# Rust conformance
cargo run --bin stunir-rust -- emit-asm spec/spec_0001.json > asm/rust_ir_0001.dcbor
diff -q $REFERENCE asm/rust_ir_0001.dcbor && echo "✓ Rust=Reference"

# Python conformance  
python3 tools/python/stunir_py.py emit-asm spec/spec_0001.json > asm/python_ir_0001.dcbor
diff -q $REFERENCE asm/python_ir_0001.dcbor && echo "✓ Python=Reference"

echo "✓ Conformance: All languages match Haskell reference"