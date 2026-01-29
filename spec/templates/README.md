# STUNIR Spec Templates

**Pipeline Stage:** spec → templates  
**Issue:** #1079

## Overview

This module provides reusable spec templates for STUNIR specifications.
Templates use variable substitution syntax `${variable}` or `${variable:default}`.

## Templates

| Template | Description |
|----------|-------------|
| `base_template.json` | Base spec structure |
| `function_template.json` | Function definition |
| `module_template.json` | Module structure |

## Usage

### CLI

```bash
# Instantiate a template
python template_engine.py base_template \
  -v id "my-spec-001" \
  -v name "My Spec" \
  -v targets '["rust", "python"]'

# Validate a template
python template_engine.py --validate base_template
```

### Python API

```python
from template_engine import TemplateEngine

engine = TemplateEngine()

# Load and instantiate
spec = engine.instantiate('base_template', {
    'id': 'my-spec-001',
    'name': 'My Spec',
    'targets': ['rust', 'python']
})

# Get deterministic hash
hash = engine.get_template_hash(spec)
```

## Variable Syntax

- `${name}` - Required variable
- `${name:default}` - Variable with default value
- Variables can be any JSON type (string, number, array, object)

## Determinism

All output uses canonical JSON (RFC 8785) with sorted keys and minimal whitespace.
