"""STUNIR Manifests Package.

This package provides manifest generation and verification for all STUNIR pipeline stages.

Phase 4 Issues:
- manifests/ir/1015
- manifests/receipts/1016
- manifests/contracts/1042
- manifests/targets/1043
- manifests/pipeline/1073
- manifests/runtime/1074
- manifests/security/1144
- manifests/performance/1145
"""

from .base import (
    canonical_json,
    compute_sha256,
    compute_file_hash,
    scan_directory,
    BaseManifestGenerator,
    BaseManifestVerifier
)

__all__ = [
    'canonical_json',
    'compute_sha256',
    'compute_file_hash',
    'scan_directory',
    'BaseManifestGenerator',
    'BaseManifestVerifier'
]
