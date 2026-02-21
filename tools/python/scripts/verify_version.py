#!/usr/bin/env python3
"""Verify version consistency across all STUNIR components."""

import re
import sys
from pathlib import Path

EXPECTED_VERSION = "0.8.9"

def check_file(filepath, pattern, description):
    """Check version in a file."""
    try:
        content = Path(filepath).read_text()
        match = re.search(pattern, content)
        if match:
            version = match.group(1)
            if version == EXPECTED_VERSION:
                print(f"✅ {description}: {version}")
                return True
            else:
                print(f"❌ {description}: {version} (expected {EXPECTED_VERSION})")
                return False
        else:
            print(f"❌ {description}: Version pattern not found")
            return False
    except FileNotFoundError:
        print(f"⚠️ {description}: File not found - {filepath}")
        return False
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False

def main():
    """Main verification function."""
    print(f"=== STUNIR Version Verification ===")
    print(f"Expected version: {EXPECTED_VERSION}\n")
    
    all_ok = True
    
    # Check pyproject.toml
    all_ok &= check_file(
        "pyproject.toml",
        r'version\s*=\s*"([^"]+)"',
        "pyproject.toml"
    )
    
    # Check src/main.rs
    all_ok &= check_file(
        "src/main.rs",
        r'\.version\("([^"]+)"\)',
        "src/main.rs"
    )
    
    # Check stunir/__init__.py
    all_ok &= check_file(
        "stunir/__init__.py",
        r'__version__: Final\[str\] = "([^"]+)"',
        "stunir/__init__.py"
    )
    
    # Check tools/rust/Cargo.toml
    all_ok &= check_file(
        "tools/rust/Cargo.toml",
        r'version\s*=\s*"([^"]+)"',
        "tools/rust/Cargo.toml"
    )
    
    print()
    if all_ok:
        print("✅ All versions are consistent!")
        return 0
    else:
        print("❌ Version inconsistencies found!")
        print(f"\nPlease update all files to version {EXPECTED_VERSION}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
