# Haskell-First Pipeline Policy (Profile-3 UN)

## Canonical Reference Implementation
Haskell binaries define correct bytes/hashes for:
- IR canonicalization (spec → dCBOR normal form, no floats)
- Receipt verification (root_attestation.dcbor + artifact digests)
- Future: Artifact emission (IR → target language)

## Confluence Contract
```
spec.json → stunir-native-hs canonicalize → ir.dcbor (SHA256: e3b0c442...)
python tools/spec_to_ir_files.py --float-policy=forbid_floats → MUST MATCH
cargo run --bin stunir-native validate → MUST MATCH
```

## Dispatch Rules (Model Only)
1. Read issues/index.machine.json → select open/priority issues
2. Haskell-first: always prefer native/haskell/ implementations  
3. Emit conformance tests for Rust/Python alignment
4. Never direct file edits—always via issues/ dispatch
