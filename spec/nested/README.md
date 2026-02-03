# STUNIR Nested Spec Resolver

**Pipeline Stage:** spec → nested  
**Issue:** #1151

## Overview

Resolves nested spec imports and produces flattened output with all
dependencies inlined.

## Files

| File | Description |
|------|-------------|
| `resolver.py` | Nested spec resolver implementation |

## Usage

### CLI

```bash
# Resolve a spec
python resolver.py spec.json

# Save flattened output
python resolver.py spec.json -o flattened.json

# Add search paths
python resolver.py spec.json -I ./modules -I ./lib

# Verbose mode
python resolver.py -v spec.json
```

### Python API

```python
from resolver import NestedResolver
from pathlib import Path

resolver = NestedResolver()

# Add search paths
resolver.add_search_path(Path('./modules'))

# Resolve from dictionary
result = resolver.resolve(spec_dict)
if result.success:
    flattened = result.flattened
    print(f"Resolved {result.imports_resolved} imports")

# Resolve from file
result = resolver.resolve_file(Path('spec.json'))
```

## Features

### Import Resolution
- Searches multiple paths for modules
- Supports various file patterns:
  - `module.json`
  - `module/index.json`
  - `module.stunir`
  - `module/spec.json`

### Circular Dependency Detection
Automatically detects and reports circular imports.

### Flattening
- Inlines all imported types, functions, and constants
- Deduplicates by name
- Sorts for deterministic output

### Import Syntax

```json
{
  "module": {
    "imports": [
      "std.io",
      {"module": "math", "items": ["sqrt", "pow"]},
      {"module": "custom", "path": "./lib/custom.json"}
    ]
  }
}
```

## Resolution Metadata

Flattened output includes `_resolved` metadata:

```json
{
  "_resolved": {
    "imports_count": 3,
    "import_order": ["std.io", "math", "custom"]
  }
}
```
