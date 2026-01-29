# STUNIR Input Validation Framework

**Pipeline Stage:** inputs → validation  
**Issue:** #1081

## Overview

Comprehensive validation framework for STUNIR input specifications.

## Files

| File | Description |
|------|-------------|
| `validator.py` | Core validation engine |
| `rules.json` | Validation rules configuration |

## Usage

### CLI

```bash
# Validate files
python validator.py spec1.json spec2.json

# JSON output
python validator.py --json spec.json

# Quiet mode (exit code only)
python validator.py -q spec.json
```

### Python API

```python
from validator import Validator, ValidationResult
from pathlib import Path

validator = Validator()

# Validate dictionary
result = validator.validate(spec_dict)
if not result.valid:
    for error in result.errors:
        print(f"{error.path}: {error.message}")

# Validate file
result = validator.validate_file(Path('spec.json'))

# Register custom validator
def check_custom_rule(spec, result, path):
    if 'custom_field' not in spec:
        result.add_warning(path, "Missing custom_field")

validator.register_validator('custom', check_custom_rule)
```

## Validation Rules

### Required Fields
- `schema` - Must match pattern `stunir.<type>.v<n>`
- `id` - Alphanumeric with underscores/hyphens
- `name` - Non-empty, max 256 characters

### Optional Validation
- `version` - Semantic versioning (X.Y.Z)
- `profile` - One of profile1-4
- `stages` - Valid pipeline stages
- `targets` - Non-empty array

### Module Validation
- Functions must have `name`
- Types must have `name` and `kind`
- Limits on counts (functions, types)

## Error Codes

| Code | Description |
|------|-------------|
| MISSING_REQUIRED | Required field missing |
| INVALID_SCHEMA | Schema format invalid |
| INVALID_ID | ID format invalid |
| INVALID_VERSION | Version not semver |
| INVALID_PROFILE | Unknown profile |
| EMPTY_TARGETS | Targets array empty |
