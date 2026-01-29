# STUNIR Visual Regression Tests

Snapshot testing for output format consistency.

## Python (pytest-snapshot)

```bash
pip install pytest-snapshot
pytest tests/visual/ -v

# Update snapshots
pytest tests/visual/ --snapshot-update
```

## Rust (insta)

```bash
cd tools/native/rust/stunir-native
cargo install cargo-insta
cargo test

# Review and update snapshots
cargo insta review
```

## What We Test

- Receipt JSON format
- Manifest JSON format
- IR output format
- Error message format
- CLI help output
