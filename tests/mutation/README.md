# STUNIR Mutation Testing

Mutation testing verifies test suite quality by introducing small changes (mutations)
to the code and checking if tests catch them.

## Python Mutation Testing (mutmut)

### Installation
```bash
pip install mutmut
```

### Running Mutation Tests
```bash
# Run full mutation testing
mutmut run

# Run on specific modules
mutmut run --paths-to-mutate=tools/ir_emitter/emit_ir.py

# View results
mutmut results

# Generate HTML report
mutmut html
```

### Configuration
See `pyproject.toml` for mutmut configuration.

## Rust Mutation Testing (cargo-mutants)

### Installation
```bash
cargo install cargo-mutants
```

### Running Mutation Tests
```bash
cd tools/native/rust/stunir-native
cargo mutants

# Run with specific options
cargo mutants --timeout 60 --jobs 4
```

## Mutation Score Thresholds

| Module | Minimum Score |
|--------|---------------|
| Core IR (tools/ir_emitter) | 80% |
| Manifests (manifests/) | 75% |
| Validation (tools/security) | 85% |
| Rust Native | 70% |

## CI Integration

Mutation testing runs weekly in CI via `.github/workflows/mutation-testing.yml`.

## Interpreting Results

- **Killed**: Mutation was detected by tests (good)
- **Survived**: Mutation was NOT detected (indicates weak tests)
- **Timeout**: Mutation caused infinite loop
- **Incompetent**: Mutation caused compilation error
