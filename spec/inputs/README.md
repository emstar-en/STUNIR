# STUNIR Input Parser

**Pipeline Stage:** spec → inputs  
**Issue:** #1048

## Overview

This module provides input parsing and validation for STUNIR spec files.

## Files

| File | Description |
|------|-------------|
| `input_schema.json` | JSON Schema for input validation |
| `parse_input.py` | Input parser implementation |

## Usage

### CLI

```bash
# Validate an input file
python parse_input.py --validate spec.json

# Parse and normalize
python parse_input.py --normalize spec.json -o output.json

# Parse only
python parse_input.py spec.json
```

### Python API

```python
from parse_input import InputParser
from pathlib import Path

parser = InputParser()

# Parse file
spec, hash_value = parser.parse_file(Path('spec.json'))

# Validate
errors = parser.validate(spec)

# Normalize with defaults
normalized = parser.normalize(spec)
```

## Schema

Required fields:
- `schema` - Version identifier (e.g., "stunir.spec.v1")
- `id` - Unique identifier
- `name` - Human-readable name

Optional fields with defaults:
- `version` - Semantic version (default: "1.0.0")
- `profile` - Profile level (default: "profile3")
- `stages` - Pipeline stages
- `targets` - Target platforms
- `dependencies` - External dependencies
- `module` - Module definition
- `metadata` - Additional metadata
