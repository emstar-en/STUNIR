# STUNIR Fuzz Testing

Fuzz testing discovers edge cases and security vulnerabilities by feeding
random/semi-random inputs to parsers and handlers.

## Python Fuzzing with Hypothesis

### Running Hypothesis Tests
```bash
pytest tests/fuzz/test_hypothesis_fuzz.py -v
```

### Writing New Fuzz Tests
```python
from hypothesis import given, strategies as st
from tests.fuzz.strategies import stunir_json_strategy

@given(stunir_json_strategy())
def test_parser_handles_any_input(data):
    # Parser should not crash on any input
    try:
        result = parser.parse(data)
        assert result is not None or result is None  # Any result is valid
    except ExpectedException:
        pass  # Known exceptions are OK
```

## Rust Fuzzing with cargo-fuzz

### Setup
```bash
cd tools/native/rust/stunir-native
cargo install cargo-fuzz
cargo +nightly fuzz init
```

### Running Fuzz Tests
```bash
cargo +nightly fuzz run fuzz_ir_parser -- -max_total_time=300
cargo +nightly fuzz run fuzz_manifest_parser -- -max_total_time=300
```

## Corpus Management

Corpus files are stored in:
- `tests/fuzz/corpus/json/` - JSON input corpus
- `tests/fuzz/corpus/ir/` - IR input corpus
- `tests/fuzz/corpus/manifest/` - Manifest input corpus

## Fuzzing Targets

| Target | Description | Priority |
|--------|-------------|----------|
| JSON Parser | Canonical JSON parsing | High |
| IR Parser | IR bundle parsing | High |
| Manifest Parser | Manifest file parsing | High |
| Path Validator | Security validation | Critical |
| Hash Functions | SHA256 computation | Medium |
