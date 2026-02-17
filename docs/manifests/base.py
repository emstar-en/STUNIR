#!/usr/bin/env python3
"""STUNIR Manifest Base Module.

Shared utilities for all manifest generators and verifiers.
Part of Phase 4: Manifests pipeline stages.
"""

import json
import hashlib
import os
import time
from typing import Dict, List, Any, Optional, Tuple


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON.
    
    Args:
        data: Data to serialize
        
    Returns:
        Canonical JSON string with sorted keys and no whitespace
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Lowercase hex digest
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to file
        
    Returns:
        Lowercase hex digest
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def scan_directory(scan_dir: str, extensions: Optional[List[str]] = None) -> List[Dict]:
    """Scan directory for files and collect metadata.
    
    Args:
        scan_dir: Directory to scan
        extensions: List of file extensions to include (e.g., ['.py', '.json'])
        
    Returns:
        List of file metadata dictionaries
    """
    files = []
    
    if not os.path.isdir(scan_dir):
        return files
    
    for root, _, filenames in os.walk(scan_dir):
        for filename in filenames:
            if extensions and not any(filename.endswith(ext) for ext in extensions):
                continue
                
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, scan_dir)
            
            try:
                file_hash = compute_file_hash(filepath)
                file_size = os.path.getsize(filepath)
                file_mtime = int(os.path.getmtime(filepath))
                
                files.append({
                    "name": filename,
                    "path": rel_path,
                    "hash": file_hash,
                    "size": file_size,
                    "mtime": file_mtime
                })
            except Exception:
                pass
    
    # Sort by path for determinism
    files.sort(key=lambda x: x['path'])
    return files


class BaseManifestGenerator:
    """Base class for manifest generators."""
    
    SCHEMA_PREFIX = "stunir.manifest"
    SCHEMA_VERSION = "v1"
    
    def __init__(self, manifest_type: str):
        """Initialize manifest generator.
        
        Args:
            manifest_type: Type of manifest (e.g., 'ir', 'receipts')
        """
        self.manifest_type = manifest_type
        self.schema = f"{self.SCHEMA_PREFIX}.{manifest_type}.{self.SCHEMA_VERSION}"
    
    def generate(self, **kwargs) -> Dict:
        """Generate manifest.
        
        Returns:
            Manifest dictionary
        """
        manifest = {
            "schema": self.schema,
            "manifest_type": self.manifest_type,
            "manifest_epoch": int(time.time()),
            "entries": [],
            "entry_count": 0
        }
        
        # Subclasses override this to populate entries
        manifest["entries"] = self._collect_entries(**kwargs)
        manifest["entry_count"] = len(manifest["entries"])
        
        # Compute manifest hash
        manifest_copy = dict(manifest)
        manifest_copy.pop('manifest_hash', None)
        manifest['manifest_hash'] = compute_sha256(canonical_json(manifest_copy))
        
        return manifest
    
    def _collect_entries(self, **kwargs) -> List[Dict]:
        """Collect entries for manifest. Override in subclasses."""
        return []
    
    def write(self, manifest: Dict, output_path: str) -> None:
        """Write manifest to file.
        
        Args:
            manifest: Manifest dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(canonical_json(manifest))


class BaseManifestVerifier:
    """Base class for manifest verifiers."""
    
    def __init__(self, manifest_type: str):
        """Initialize manifest verifier.
        
        Args:
            manifest_type: Type of manifest (e.g., 'ir', 'receipts')
        """
        self.manifest_type = manifest_type
    
    def verify(self, manifest_path: str, **kwargs) -> Tuple[bool, List[str], List[str], Dict]:
        """Verify manifest.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            Tuple of (is_valid, errors, warnings, stats)
        """
        errors = []
        warnings = []
        stats = {'verified': 0, 'missing': 0, 'hash_mismatch': 0}
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            return False, [f"Failed to load manifest: {e}"], [], stats
        
        # Check schema
        schema = manifest.get('schema', '')
        if not schema.startswith(f'stunir.manifest.{self.manifest_type}'):
            errors.append(f"Invalid schema: {schema}")
        
        # Verify manifest hash
        manifest_copy = dict(manifest)
        stored_hash = manifest_copy.pop('manifest_hash', None)
        computed_hash = compute_sha256(canonical_json(manifest_copy))
        
        if stored_hash and stored_hash != computed_hash:
            errors.append(f"Manifest hash mismatch: stored={stored_hash}, computed={computed_hash}")
        
        # Subclass-specific verification
        sub_errors, sub_warnings, sub_stats = self._verify_entries(manifest, **kwargs)
        errors.extend(sub_errors)
        warnings.extend(sub_warnings)
        stats.update(sub_stats)
        
        return len(errors) == 0, errors, warnings, stats
    
    def _verify_entries(self, manifest: Dict, **kwargs) -> Tuple[List[str], List[str], Dict]:
        """Verify manifest entries. Override in subclasses."""
        return [], [], {}
