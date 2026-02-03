# STUNIR Integration Tests

This directory contains end-to-end integration tests for the STUNIR pipeline.

## Test Structure

```
tests/integration/
├── __init__.py              # Package marker
├── conftest.py              # pytest fixtures
├── utils.py                 # Test utilities
├── test_complete_pipeline.py # Full pipeline tests
├── test_receipt_verification.py # Receipt tests
└── test_multi_target.py     # Multi-target tests
```

## Running Tests

### Run All Integration Tests

```bash
pytest tests/integration/ -v
```

### Run Specific Test File

```bash
pytest tests/integration/test_complete_pipeline.py -v
```

### Run Single Test

```bash
pytest tests/integration/test_complete_pipeline.py::TestCompletePipeline::test_spec_to_ir_conversion -v
```

### Run with Coverage

```bash
pytest tests/integration/ -v --cov=tools --cov=manifests
```

## Test Categories

### 1. Complete Pipeline Tests (`test_complete_pipeline.py`)

Tests the full STUNIR pipeline:
- Spec → IR conversion
- IR → Manifest generation
- Full pipeline determinism
- Multi-module spec processing

### 2. Receipt Verification Tests (`test_receipt_verification.py`)

Tests receipt generation and verification:
- Receipt creation
- Multi-artifact receipts
- Receipt determinism
- Receipt verification (valid/invalid)
- Receipt persistence (save/load)

### 3. Multi-Target Tests (`test_multi_target.py`)

Tests generating multiple targets:
- Python, Bash, JavaScript targets
- Function name preservation
- Target generation determinism
- Target consistency
- Manifest tracking

## Fixtures

Available pytest fixtures (from `conftest.py`):

| Fixture | Description |
|---------|-------------|
| `project_root` | Path to project root directory |
| `temp_dir` | Temporary directory (auto-cleaned) |
| `sample_spec` | Sample STUNIR spec dictionary |
| `sample_ir` | Sample STUNIR IR dictionary |
| `sample_manifest` | Sample manifest dictionary |
| `spec_file` | Spec file in temp directory |
| `ir_file` | IR file in temp directory |
| `tools_available` | Dict of available tools |

## Utilities

Available utilities (from `utils.py`):

```python
from tests.integration.utils import (
    compute_sha256,          # Hash a string
    compute_file_sha256,     # Hash a file
    canonical_json,          # Canonical JSON output
    load_json_file,          # Load JSON from file
    save_json_file,          # Save JSON to file
    verify_manifest_structure,  # Validate manifest
    verify_ir_structure,     # Validate IR
    assert_deterministic,    # Assert function is deterministic
    create_test_spec,        # Create test spec
    create_test_ir,          # Create test IR
)
```

## Writing New Tests

### 1. Basic Test Structure

```python
import pytest
from pathlib import Path
from .utils import create_test_ir, compute_sha256

class TestMyFeature:
    """Test my feature."""

    def test_basic_functionality(self, temp_dir: Path):
        """Test basic feature functionality."""
        # Arrange
        ir = create_test_ir("test", functions=2)

        # Act
        result = my_feature(ir)

        # Assert
        assert result is not None
        assert len(result) > 0
```

### 2. Testing Determinism

```python
def test_determinism(self):
    """Test output is deterministic."""
    ir = create_test_ir("test")

    results = [process(ir) for _ in range(3)]

    first = results[0]
    for r in results[1:]:
        assert r == first, "Output should be deterministic"
```

### 3. Testing with Files

```python
def test_file_operations(self, temp_dir: Path):
    """Test with file operations."""
    # Create test file
    test_file = temp_dir / "test.json"
    save_json_file(test_file, {"data": "test"})

    # Process file
    result = process_file(test_file)

    # Verify output
    output_file = temp_dir / "output.json"
    assert output_file.exists()
```

## Best Practices

### Do

- ✅ Use fixtures for common setup
- ✅ Test determinism for all hashing/serialization
- ✅ Use temporary directories for file operations
- ✅ Test both success and failure cases
- ✅ Use clear, descriptive test names
- ✅ Keep tests independent

### Don't

- ❌ Don't depend on external services
- ❌ Don't use hardcoded paths outside temp dirs
- ❌ Don't skip error case testing
- ❌ Don't write tests that depend on order
- ❌ Don't leave test files behind

## Test Data Management

### Creating Test Data

```python
# Use utility functions
spec = create_test_spec("my-spec", modules=3)
ir = create_test_ir("my-ir", functions=5)
```

### Using Fixtures

```python
def test_with_fixtures(self, sample_spec, sample_ir, temp_dir):
    # sample_spec and sample_ir are pre-configured
    save_json_file(temp_dir / "spec.json", sample_spec)
```

### Test Vectors

For complex test data, create JSON files in `test_vectors/`:

```python
from pathlib import Path

def load_test_vector(name: str) -> dict:
    path = Path(__file__).parent.parent.parent / "test_vectors" / name
    return load_json_file(path)
```

## CI Integration

Integration tests run in GitHub Actions:
- On push to main/devsite
- On pull requests
- Python 3.10, 3.11, 3.12
- Ubuntu and macOS

See `.github/workflows/ci.yml` for configuration.
