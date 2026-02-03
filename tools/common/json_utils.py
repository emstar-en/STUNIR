"""STUNIR JSON Utilities - Canonical JSON Serialization.

Provides deterministic JSON output following RFC 8785 (JSON Canonicalization Scheme).
All JSON output in STUNIR should use these utilities to ensure reproducibility.
"""

import json
from typing import Any, Optional, Union, IO


def canonical_json(data: Any) -> str:
    """Generate RFC 8785 / JCS subset canonical JSON.
    
    Features:
    - Sorted keys (alphabetically)
    - No unnecessary whitespace
    - UTF-8 encoding
    - Consistent number formatting
    
    Args:
        data: Any JSON-serializable Python object
        
    Returns:
        Canonical JSON string representation
        
    Raises:
        TypeError: If data contains non-serializable types
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def canonical_json_bytes(data: Any) -> bytes:
    """Generate canonical JSON as UTF-8 bytes.
    
    Args:
        data: Any JSON-serializable Python object
        
    Returns:
        UTF-8 encoded canonical JSON bytes
    """
    return canonical_json(data).encode('utf-8')


def canonical_json_pretty(data: Any, indent: int = 2) -> str:
    """Generate pretty-printed canonical JSON (sorted keys).
    
    While not strictly canonical (contains whitespace), this maintains
    key ordering for human readability.
    
    Args:
        data: Any JSON-serializable Python object
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        Pretty-printed JSON string with sorted keys
    """
    return json.dumps(data, sort_keys=True, indent=indent, ensure_ascii=False)


def parse_json(content: Union[str, bytes], strict: bool = True) -> Any:
    """Parse JSON content safely.
    
    Args:
        content: JSON string or bytes to parse
        strict: If True, raise on invalid JSON; if False, return None
        
    Returns:
        Parsed Python object, or None if strict=False and parsing fails
        
    Raises:
        json.JSONDecodeError: If strict=True and content is invalid JSON
    """
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if strict:
            raise
        return None


def json_load_file(filepath: str, strict: bool = True) -> Any:
    """Load and parse JSON from a file.
    
    Args:
        filepath: Path to JSON file
        strict: If True, raise on errors; if False, return None
        
    Returns:
        Parsed Python object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If strict=True and content is invalid JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        if strict:
            raise
        return None


def json_dump_file(data: Any, filepath: str, pretty: bool = False) -> None:
    """Write JSON data to a file.
    
    Args:
        data: Data to serialize
        filepath: Output file path
        pretty: If True, use pretty printing; otherwise use canonical format
    """
    content = canonical_json_pretty(data) if pretty else canonical_json(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write('\n')  # Trailing newline for POSIX compliance
