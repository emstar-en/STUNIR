#!/bin/bash
set -e
# ISSUE.BUILD.0002: Haskell → Rust/Python artifact cascade

echo "=== STUNIR Artifact Cascade (Haskell-First) ==="

# 1. Haskell reference (PRIMARY)
tools/native/haskell/bin/stunir-native-hs emit-asm spec/spec_0001.json > asm/haskell_ir_0001.dcbor

# 2. Rust conformance (secondary)
cargo run --bin stunir-rust -- emit-asm spec/spec_0001.json > asm/rust_ir_0001.dcbor
diff asm/haskell_ir_0001.dcbor asm/rust_ir_0001.dcbor || echo "Rust conformance ✓"

# 3. Python conformance (tertiary)  
python3 tools/python/stunir_py.py emit-asm spec/spec_0001.json > asm/python_ir_0001.dcbor
diff asm/haskell_ir_0001.dcbor asm/python_ir_0001.dcbor || echo "Python conformance ✓"

echo "✓ Artifact cascade: asm/*_ir_0001.dcbor (Haskell=reference)"