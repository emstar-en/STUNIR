#!/usr/bin/env python3
"""STUNIR DO-332 OOP Verification Python Wrapper.

This is a minimal wrapper that invokes the Ada SPARK binary.
All logic is implemented in SPARK.

Copyright (C) 2026 STUNIR Project
SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


class DO332Analyzer:
    """Wrapper for DO-332 OOP verification analyzer."""

    def __init__(self, binary_path: Optional[str] = None):
        """Initialize the analyzer.

        Args:
            binary_path: Path to do332_analyzer binary.
                        If None, searches in standard locations.
        """
        if binary_path:
            self.binary = Path(binary_path)
        else:
            self.binary = self._find_binary()

    def _find_binary(self) -> Path:
        """Find the do332_analyzer binary."""
        # Check relative paths
        candidates = [
            Path(__file__).parent / "bin" / "do332_analyzer",
            Path(__file__).parent.parent.parent / "bin" / "do332_analyzer",
            Path("/home/ubuntu/stunir_repo/tools/do332/bin/do332_analyzer"),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # Check PATH
        import shutil
        found = shutil.which("do332_analyzer")
        if found:
            return Path(found)

        # Return default (will fail if not built)
        return Path(__file__).parent / "bin" / "do332_analyzer"

    def analyze(
        self,
        ir_dir: str = "asm/ir",
        output_dir: str = "receipts/do332",
        dal: str = "C",
        analyses: str = "all",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run DO-332 analysis.

        Args:
            ir_dir: Directory containing IR files.
            output_dir: Directory for output reports.
            dal: DAL level (A, B, C, D, E).
            analyses: Comma-separated list or 'all'.
            verbose: Enable verbose output.

        Returns:
            Dictionary containing analysis results.
        """
        args = [
            str(self.binary),
            "--ir-dir", ir_dir,
            "--output-dir", output_dir,
            "--dal", dal,
            "--analyses", analyses,
        ]

        if verbose:
            args.append("--verbose")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_dir": output_dir,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "error": "Analysis timed out",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "returncode": -1,
                "error": f"Binary not found: {self.binary}",
                "hint": "Run 'make build' to build the analyzer",
            }

    def version(self) -> str:
        """Get analyzer version."""
        try:
            result = subprocess.run(
                [str(self.binary), "--version"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return "unknown"

    def is_available(self) -> bool:
        """Check if the analyzer binary is available."""
        return self.binary.exists() and self.binary.is_file()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DO-332 OOP Verification Analyzer (Python Wrapper)"
    )
    parser.add_argument(
        "--ir-dir",
        default="asm/ir",
        help="Input IR directory",
    )
    parser.add_argument(
        "--output-dir",
        default="receipts/do332",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--dal",
        default="C",
        choices=["A", "B", "C", "D", "E"],
        help="DAL level",
    )
    parser.add_argument(
        "--analyses",
        default="all",
        help="Analyses to run (all, inheritance, polymorphism, dispatch, coupling)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if analyzer is available",
    )

    args = parser.parse_args()

    analyzer = DO332Analyzer()

    if args.version:
        print(f"DO-332 Wrapper version 1.0.0")
        print(f"Analyzer: {analyzer.version()}")
        return 0

    if args.check:
        if analyzer.is_available():
            print(f"Analyzer available: {analyzer.binary}")
            return 0
        else:
            print(f"Analyzer not found. Run 'make build' in tools/do332/")
            return 1

    # Run analysis
    result = analyzer.analyze(
        ir_dir=args.ir_dir,
        output_dir=args.output_dir,
        dal=args.dal,
        analyses=args.analyses,
        verbose=args.verbose,
    )

    if result["success"]:
        print(result.get("stdout", ""))
        return 0
    else:
        print(f"Error: {result.get('error', result.get('stderr', 'Unknown error'))}",
              file=sys.stderr)
        if "hint" in result:
            print(f"Hint: {result['hint']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
