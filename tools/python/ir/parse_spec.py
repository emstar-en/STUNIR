#!/usr/bin/env python3
"""Parse specification files to Semantic IR.

This is the main CLI tool for converting high-level specifications
to Semantic IR format.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

from .parser import SpecParser
from .types import ParserOptions


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Parse specification files to Semantic IR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse embedded specification
  %(prog)s --input spec.json --category embedded --output ir.json
  
  # Parse with validation
  %(prog)s --input spec.yaml --category gpu --validate
  
  # Pretty print output
  %(prog)s --input spec.json --category lisp --format pretty
  
  # Verbose mode
  %(prog)s --input spec.json --category wasm --verbose
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input specification file (JSON or YAML)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output Semantic IR file (default: stdout)"
    )
    
    parser.add_argument(
        "--category", "-c",
        required=True,
        help="Target category (embedded, gpu, lisp, prolog, etc.)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "pretty", "compact"],
        default="pretty",
        help="Output format (default: pretty)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated IR"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (print errors and warnings)"
    )
    
    parser.add_argument(
        "--no-type-inference",
        action="store_true",
        help="Disable type inference"
    )
    
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Maximum number of errors before stopping (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Create parser options
    options = ParserOptions(
        category=args.category,
        validate_schema=True,
        collect_metrics=True,
        enable_type_inference=not args.no_type_inference,
        max_errors=args.max_errors,
    )
    
    # Create parser
    if args.verbose:
        print(f"Parsing {args.input} as {args.category} category...")
    
    spec_parser = SpecParser(options)
    
    # Parse specification
    ir = spec_parser.parse_file(str(input_path))
    
    # Check for errors
    errors = spec_parser.get_errors()
    if errors:
        print(f"\\nParsing errors ({len(errors)}):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        
        if not ir:
            print("\\nFatal: Parsing failed", file=sys.stderr)
            return 1
    
    # Validate if requested
    if args.validate:
        if args.verbose:
            print("Validating IR...")
        
        result = spec_parser.validate_ir(ir)
        if not result.is_valid:
            print(f"\\nValidation errors ({len(result.errors)}):", file=sys.stderr)
            for error in result.errors:
                print(f"  {error}", file=sys.stderr)
            return 1
        
        if args.verbose:
            print("✓ IR validation passed")
    
    # Format output
    if args.format == "compact":
        output = ir.to_json(pretty=False)
    else:
        output = ir.to_json(pretty=True)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        if args.verbose:
            print(f"Writing output to {args.output}...")
        
        with open(output_path, 'w') as f:
            f.write(output)
        
        if args.verbose:
            print(f"✓ Generated Semantic IR: {output_path}")
            print(f"  Functions: {len(ir.functions)}")
            print(f"  Types: {len(ir.types)}")
            print(f"  Constants: {len(ir.constants)}")
    else:
        print(output)
    
    # Print summary if verbose
    if args.verbose and errors:
        print(f"\\n⚠ Completed with {len(errors)} warnings")
    elif args.verbose:
        print("\\n✓ Parsing completed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
