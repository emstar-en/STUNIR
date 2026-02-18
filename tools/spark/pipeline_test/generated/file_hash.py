#!/usr/bin/env python3
"""
file_hash.py - SHA-256 File Hasher
Cleanroom implementation based on behavioral specification from Ada source

Specification:
- Compute SHA-256 hash of files
- Output: lowercase hexadecimal (64 characters)
- Exit codes: 0=success, 1=file not found, 2=usage error, 3=IO error
"""

import sys
import hashlib

def compute_hash(file_path):
    """Compute SHA-256 hash of a file"""
    try:
        hasher = hashlib.sha256()
        chunk_size = 8192  # 8KB chunks, matching Ada implementation
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return hasher.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied: {file_path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)

def main():
    if len(sys.argv) < 2:
        print("Usage: file_hash.py [--describe] <file_path>", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  --describe    Show metadata about the hash computation", file=sys.stderr)
        sys.exit(2)
    
    describe_mode = False
    file_path = None
    
    # Parse arguments
    for arg in sys.argv[1:]:
        if arg == "--describe":
            describe_mode = True
        elif arg.startswith("--"):
            print(f"Error: Unknown option: {arg}", file=sys.stderr)
            sys.exit(2)
        else:
            file_path = arg
    
    if file_path is None:
        print("Error: Missing file path", file=sys.stderr)
        sys.exit(2)
    
    # Compute hash
    hash_value = compute_hash(file_path)
    
    # Validate hash length
    if len(hash_value) != 64:
        print("Error: Invalid hash format", file=sys.stderr)
        sys.exit(3)
    
    # Output hash
    print(hash_value)
    
    # Describe mode output
    if describe_mode:
        print("--describe output:")
        print("Algorithm: SHA-256")
        print("Hash length: 64 characters")
        print("Format: Lowercase hexadecimal")
        print(f"File processed: {file_path}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
