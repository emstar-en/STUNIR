# STUNIR Spec Examples

**Pipeline Stage:** spec → examples  
**Issue:** #1080

## Overview

This directory contains example STUNIR spec files demonstrating
various features and patterns.

## Examples

| File | Description |
|------|-------------|
| `minimal.json` | Minimal valid spec with required fields only |
| `with_functions.json` | Spec with function definitions |
| `with_types.json` | Spec with custom type definitions |
| `full_module.json` | Complete module with all features |

## Validation

Validate all examples:

```bash
python validate_examples.py
```

Validate specific files:

```bash
python validate_examples.py minimal.json with_types.json
```

## Spec Structure

```json
{
  "schema": "stunir.spec.v1",
  "id": "unique-id",
  "name": "Human Readable Name",
  "version": "1.0.0",
  "profile": "profile3",
  "stages": ["STANDARDIZATION", "UNIQUE_NORMALS", "IR", "BINARY", "RECEIPT"],
  "targets": ["rust", "c99"],
  "module": {
    "name": "module_name",
    "types": [...],
    "functions": [...]
  }
}
```

## Using as Templates

These examples can serve as starting points for new specs.
Copy and modify as needed, or use with the template engine:

```python
from spec.templates.template_engine import TemplateEngine

engine = TemplateEngine()
base = engine.load_template('base_template')
```
