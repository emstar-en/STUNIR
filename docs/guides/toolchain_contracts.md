# Toolchain contracts (lifted toolchains)

This directory documents how STUNIR "lifts" existing toolchains (runtimes, compilers, engines) into trackable and attestable operations.

## The pattern

1. Add a contract JSON in `contracts/*.json`.
2. Map a target name to required contracts in `contracts/target_requirements.json`.
3. Add one or more deterministic test vectors in `test_vectors/<lang>/`.
4. Probe tools to produce acceptance receipts:

- `scripts/ensure_deps.sh` emits:
  - `receipts/requirements.json`
  - `receipts/deps/<contract>.json`

5. Verify acceptance receipts satisfy requirements:

- `scripts/verify_deps.sh`

## Notes

- Contracts should prefer **file-producing determinism tests** (same inputs => byte-identical outputs).
- For ecosystems that risk network access during probe (e.g., `dotnet restore`), start with identity-only contracts and strengthen later with offline test vectors.
