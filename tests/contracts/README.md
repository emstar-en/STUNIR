# STUNIR Contract Tests

Contract testing ensures API and format compatibility.

## Running Contract Tests

```bash
pytest tests/contracts/ -v
```

## Contract Types

### API Contracts
- Python module APIs (function signatures, return types)
- CLI contracts (arguments, exit codes)

### Format Contracts
- IR JSON schema
- Manifest JSON schema
- Receipt JSON schema

## Schema Validation

Uses JSON Schema for format validation.
