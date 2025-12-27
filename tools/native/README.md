# STUNIR Native Stages (Rust + Haskell)

This bundle adds **Python-free** native implementations for:

1. **IR Validation** (Standardization gate)
2. **Pack / Receipt Verification** (Profile-3-style pack verifier)

It ships **two equivalent CLIs**:

- `tools/native/rust/stunir-native` (Cargo)
- `tools/native/haskell/stunir-native` (Cabal)

Both aim to match the failure tags in `stunir_profile3_contract.json` for pack verification (exit 1 + stable tag string).

## Commands

- `stunir-native validate <ir.json>`
- `stunir-native verify pack ...`
- `stunir-native verify emit <stunir.emit.v1.json>`

### Canonical JSON note (numbers)

Both canonicalizers intentionally **reject non-integer JSON numbers**. Extend to full RFC 8785 number formatting if IR later uses floats.
