"""STUNIR - Deterministic IR Generation and Verification Toolkit.

STUNIR provides tools for generating and verifying deterministic
Intermediate Representation (IR) bundles with manifest verification.

Example:
    >>> import stunir
    >>> print(stunir.__version__)
    '1.0.0'

Attributes:
    __version__: The current STUNIR version.
    API_VERSION: The API version for compatibility checking.
"""

from typing import Final

__version__: Final[str] = "1.0.0"
__author__: Final[str] = "STUNIR Team"
__license__: Final[str] = "MIT"

# API version for compatibility checking
API_VERSION: Final[str] = "1"

# Schema versions
IR_SCHEMA_VERSION: Final[str] = "stunir.ir.v1"
MANIFEST_SCHEMA_VERSION: Final[str] = "stunir.manifest.v1"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "API_VERSION",
    "IR_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
]
