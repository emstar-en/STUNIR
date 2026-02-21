#!/usr/bin/env python3
"""STUNIR Toolchain Manifest Generator.

This tool is part of the tools → manifest pipeline stage.
It generates toolchain manifests for STUNIR builds.

Usage:
    gen_manifest.py [--output=<file>] [--scan-dir=<dir>]
"""

import json
import sys
import hashlib
import os
import time
import subprocess
from typing import Any, Dict, List, Optional

def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_tool_version(tool_path: str) -> str:
    """Try to get version of a tool.

    Args:
        tool_path: Path to the tool executable

    Returns:
        Version string or "unknown" if version cannot be determined
    """
    try:
        result = subprocess.run(
            [tool_path, '--version'],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False  # SECURITY: Never use shell=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError) as e:
        # Log specific error type for debugging
        import logging
        logging.debug(f"Could not get version for {tool_path}: {type(e).__name__}: {e}")
    return "unknown"

def scan_tools(scan_dir: str) -> List[Dict[str, Any]]:
    """Scan directory for tools and collect metadata."""
    tools: List[Dict[str, Any]] = []
    
    for root, dirs, files in os.walk(scan_dir):
        for file in files:
            if file.endswith('.py') or file.endswith('.sh'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, scan_dir)
                
                try:
                    file_hash = compute_file_hash(filepath)
                    file_size = os.path.getsize(filepath)
                    
                    tools.append({
                        "name": os.path.splitext(file)[0],
                        "path": rel_path,
                        "hash": file_hash,
                        "size": file_size,
                        "type": "python" if file.endswith('.py') else "shell"
                    })
                except Exception as e:
                    print(f"Warning: Could not process {filepath}: {e}", file=sys.stderr)
    
    # Sort by name for determinism
    tools.sort(key=lambda x: x['name'])
    return tools

def generate_manifest(scan_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate toolchain manifest."""
    manifest = {
        "schema": "stunir.manifest.toolchain.v1",
        "manifest_epoch": int(time.time()),
        "manifest_tools": [],
        "manifest_count": 0
    }

    if scan_dir and os.path.isdir(scan_dir):
        manifest["manifest_tools"] = scan_tools(scan_dir)
        manifest["manifest_count"] = len(manifest["manifest_tools"])

    # Compute manifest hash (excluding the hash field itself)
    manifest_copy = dict(manifest)
    manifest_copy.pop('manifest_hash', None)
    manifest['manifest_hash'] = compute_sha256(canonical_json(manifest_copy))

    return manifest

def parse_args(argv: list[str]) -> Dict[str, Optional[str]]:
    """Parse command line arguments."""
    args = {'output': None, 'scan_dir': None}

    for arg in argv[1:]:
        if arg.startswith('--output='):
            args['output'] = arg.split('=', 1)[1]
        elif arg.startswith('--scan-dir='):
            args['scan_dir'] = arg.split('=', 1)[1]

    return args

def main() -> None:
    """Generate a toolchain manifest from the command line."""
    args = parse_args(sys.argv)
    
    try:
        # Generate manifest
        manifest = generate_manifest(args['scan_dir'])
        
        # Canonicalize
        canonical_output = canonical_json(manifest)
        
        # Output
        if args['output']:
            with open(args['output'], 'w') as f:
                f.write(canonical_output)
            print(f"Manifest written to {args['output']}", file=sys.stderr)
        else:
            print(canonical_output)
        
        print(f"Tools found: {manifest['manifest_count']}", file=sys.stderr)
        print(f"Manifest hash: {manifest['manifest_hash']}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
