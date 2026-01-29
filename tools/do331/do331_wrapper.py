#!/usr/bin/env python3
"""STUNIR DO-331 Model-Based Development Wrapper

This is a MINIMAL Python wrapper that calls the Ada SPARK binary.
All logic is implemented in SPARK - this wrapper only provides a
convenient Python interface for integration.

Copyright (C) 2026 STUNIR Project
SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any


class DO331Wrapper:
    """Wrapper for DO-331 SPARK tools."""
    
    def __init__(self, binary_path: Optional[str] = None):
        """Initialize wrapper.
        
        Args:
            binary_path: Path to do331_main binary. If None, searches
                        standard locations.
        """
        if binary_path:
            self.binary = Path(binary_path)
        else:
            # Search standard locations
            search_paths = [
                Path(__file__).parent / "bin" / "do331_main",
                Path(__file__).parent.parent.parent / "bin" / "do331_main",
                Path("/home/ubuntu/stunir_repo/tools/do331/bin/do331_main"),
            ]
            for p in search_paths:
                if p.exists():
                    self.binary = p
                    break
            else:
                self.binary = None
    
    def is_available(self) -> bool:
        """Check if the SPARK binary is available."""
        return self.binary is not None and self.binary.exists()
    
    def run(self, *args: str) -> subprocess.CompletedProcess:
        """Run the DO-331 tool with arguments.
        
        Args:
            *args: Command line arguments.
            
        Returns:
            CompletedProcess with stdout/stderr.
        """
        if not self.is_available():
            raise RuntimeError(
                "DO-331 binary not found. Build with 'make build' first."
            )
        
        cmd = [str(self.binary)] + list(args)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
    
    def self_test(self) -> bool:
        """Run self-test.
        
        Returns:
            True if all tests pass.
        """
        result = self.run("--test")
        return result.returncode == 0 and "FAIL" not in result.stdout
    
    def get_version(self) -> str:
        """Get tool version."""
        result = self.run("--version")
        return result.stdout.strip()
    
    def transform(
        self,
        ir_input: str,
        output_file: str,
        dal_level: str = "C",
        include_coverage: bool = True
    ) -> Dict[str, Any]:
        """Transform IR to SysML 2.0 model.
        
        Note: This is a placeholder. The full implementation would
        call the SPARK binary with appropriate arguments.
        
        Args:
            ir_input: Path to IR JSON file.
            output_file: Path for output SysML file.
            dal_level: DAL level (A, B, C, D, E).
            include_coverage: Whether to include coverage points.
            
        Returns:
            Result dictionary with status and metadata.
        """
        # For now, return a status indicating the SPARK binary should be used
        return {
            "status": "not_implemented",
            "message": "Use the SPARK binary directly: ./bin/do331_main --transform",
            "dal_level": dal_level,
            "include_coverage": include_coverage
        }


def main():
    """Command-line interface."""
    wrapper = DO331Wrapper()
    
    if len(sys.argv) < 2:
        print("STUNIR DO-331 Python Wrapper")
        print("")
        print("This is a thin wrapper around the Ada SPARK binary.")
        print("For full functionality, use the SPARK binary directly.")
        print("")
        print(f"Binary available: {wrapper.is_available()}")
        if wrapper.is_available():
            print(f"Binary path: {wrapper.binary}")
            print("")
            print("Run: ./bin/do331_main --help")
        else:
            print("")
            print("Build the binary with: make build")
        return
    
    # Pass through to binary
    if wrapper.is_available():
        result = wrapper.run(*sys.argv[1:])
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    else:
        print("ERROR: DO-331 binary not available.", file=sys.stderr)
        print("Build with: cd tools/do331 && make build", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
