#!/usr/bin/env python3
"""STUNIR DO-333 Formal Methods Wrapper

Minimal Python wrapper for DO-333 SPARK tools.
All logic is implemented in Ada SPARK - this is just an interface.

Copyright (C) 2026 STUNIR Project
SPDX-License-Identifier: Apache-2.0
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Tool paths
SCRIPT_DIR = Path(__file__).parent.resolve()
BIN_DIR = SCRIPT_DIR / "bin"
DO333_ANALYZER = BIN_DIR / "do333_analyzer"


def check_compliance_enabled() -> bool:
    """Check if STUNIR compliance tools are enabled."""
    return os.environ.get("STUNIR_ENABLE_COMPLIANCE", "0") == "1"


def check_tool_built() -> bool:
    """Check if DO-333 analyzer is built."""
    return DO333_ANALYZER.exists() and os.access(DO333_ANALYZER, os.X_OK)


def run_analyzer(args: list) -> int:
    """Run the DO-333 analyzer with given arguments."""
    if not check_compliance_enabled():
        print("DO-333 support disabled. Set STUNIR_ENABLE_COMPLIANCE=1")
        return 0
    
    if not check_tool_built():
        print(f"DO-333 analyzer not found at {DO333_ANALYZER}")
        print("Run 'make -C tools/do333' to build.")
        return 1
    
    try:
        result = subprocess.run(
            [str(DO333_ANALYZER)] + args,
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running analyzer: {e}")
        return 1


def demo() -> int:
    """Run DO-333 demonstration."""
    return run_analyzer(["demo"])


def analyze(project_file: str) -> int:
    """Analyze a project for formal verification."""
    return run_analyzer(["analyze", project_file])


def generate_report(format_type: str = "text") -> int:
    """Generate certification report."""
    return run_analyzer(["report", format_type])


def generate_matrix() -> int:
    """Generate DO-333 compliance matrix."""
    return run_analyzer(["matrix"])


def get_version() -> str:
    """Get tool version."""
    if not check_tool_built():
        return "DO-333 Tools (not built)"
    
    try:
        result = subprocess.run(
            [str(DO333_ANALYZER), "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout.strip()
    except Exception:
        return "DO-333 Tools (unknown version)"


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("STUNIR DO-333 Formal Methods Wrapper")
        print("")
        print("Usage: do333_wrapper.py <command> [options]")
        print("")
        print("Commands:")
        print("  demo              Run demonstration")
        print("  analyze <gpr>     Analyze project")
        print("  report [format]   Generate report")
        print("  matrix            Generate compliance matrix")
        print("  --version         Show version")
        print("  --help            Show help")
        return 0
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    if cmd == "--help" or cmd == "-h":
        return main()  # Show help
    elif cmd == "--version" or cmd == "-v":
        print(get_version())
        return 0
    elif cmd == "demo":
        return demo()
    elif cmd == "analyze":
        if not args:
            print("Error: No project file specified")
            return 1
        return analyze(args[0])
    elif cmd == "report":
        fmt = args[0] if args else "text"
        return generate_report(fmt)
    elif cmd == "matrix":
        return generate_matrix()
    else:
        print(f"Unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
