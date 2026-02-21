# STUNIR Manifest Tools

Part of the `tools → manifest` pipeline stage.

## Overview

Manifest tools generate and verify toolchain manifests.

## Tools

### gen_manifest.py

Generates toolchain manifest.

```bash
python gen_manifest.py [--output=<file>] [--scan-dir=<dir>]
```

Options:
- `--output=<file>`: Output file (default: stdout)
- `--scan-dir=<dir>`: Directory to scan for tools

### verify_manifest.py

Verifies manifest against actual tools.

```bash
python verify_manifest.py <manifest.json> [--tools-dir=<dir>]
```

## Output Schema

```json
{
  "schema": "stunir.manifest.toolchain.v1",
  "manifest_epoch": 1735500000,
  "manifest_tools": [
    {
      "name": "tool_name",
      "path": "relative/path/to/tool.py",
      "hash": "sha256...",
      "size": 1234,
      "type": "python"
    }
  ],
  "manifest_count": 1,
  "manifest_hash": "sha256..."
}
```

## Determinism

Manifests are canonical JSON with sorted keys and tools.
