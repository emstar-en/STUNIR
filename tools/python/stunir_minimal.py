#!/usr/bin/env python3
"""Minimal Python implementation of core STUNIR toolchain commands."""
import argparse
import json
import sys
import hashlib
from typing import Any, Dict, List

# --- Data Structures (Implicit via Dicts, but logic mirrors Native) ---


def create_ir_function(name: str, body: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create an IR function record.

    Args:
        name: The function name.
        body: List of IR instructions forming the function body.

    Returns:
        A dictionary representing an IR function with name and body.
    """
    # Sort keys to ensure deterministic output if we were constructing raw dicts
    # But here we just return the structure.
    return {"name": name, "body": body}


def create_ir_instruction(op: str, args: List[str]) -> Dict[str, Any]:
    """Create an IR instruction record.

    Args:
        op: The operation code (e.g., "LOAD", "STORE").
        args: List of argument strings for the operation.

    Returns:
        A dictionary representing an IR instruction with operation and arguments.
    """
    return {"op": op, "args": args}


# --- Commands ---


def cmd_spec_to_ir(in_json: str, out_ir: str) -> None:
    """Convert a spec JSON file into a minimal IR JSON file.

    Args:
        in_json: Path to input specification JSON file.
        out_ir: Path to output IR JSON file.

    Raises:
        SystemExit: If reading the spec or writing the IR fails.
    """
    try:
        with open(in_json, 'r') as f:
            _spec = json.load(f)
    except Exception as e:
        sys.exit(f"Failed to read spec: {e}")

    # Mock Transformation (Matching Rust/Haskell Stub)
    main_fn = create_ir_function("main", [
        create_ir_instruction("LOAD", ["r1", "0"]),
        create_ir_instruction("STORE", ["r1", "result"])
    ])

    ir = {
        "version": "1.0.0",
        "functions": [main_fn]
    }

    # Write IR (Deterministic: indent=2, sort_keys=True)
    try:
        with open(out_ir, 'w') as f:
            json.dump(ir, f, indent=2, sort_keys=True)
            # Add newline to match some pretty printers if needed,
            # but standard json.dump usually suffices for logic.
        print(f"Generated IR at: {out_ir}")
    except Exception as e:
        sys.exit(f"Failed to write IR: {e}")


def cmd_gen_provenance(in_ir: str, epoch_json: str, out_prov: str) -> None:
    """Generate a minimal provenance record for an IR file.

    Args:
        in_ir: Path to input IR JSON file.
        epoch_json: Path to epoch JSON file containing epoch metadata.
        out_prov: Path to output provenance JSON file.

    Raises:
        SystemExit: If reading inputs or writing the provenance fails.
    """
    # 1. Read IR
    try:
        with open(in_ir, 'rb') as f:
            ir_bytes = f.read()
            # In Python, we must be careful. If we read as text and json.load,
            # we lose the exact byte representation for hashing.
            # However, the Native tools hash the FILE content.
            ir_hash = hashlib.sha256(ir_bytes).hexdigest()
    except Exception as e:
        sys.exit(f"Failed to read IR: {e}")

    # 2. Read Epoch
    try:
        with open(epoch_json, 'r') as f:
            epoch_data = json.load(f)
    except Exception as e:
        sys.exit(f"Failed to read Epoch: {e}")

    # 3. Create Provenance
    prov = {
        "epoch": epoch_data,
        "ir_hash": f"sha256:{ir_hash}",
        "schema": "stunir.provenance.v1",
        "status": "SUCCESS"
    }

    # 4. Write Output
    try:
        with open(out_prov, 'w') as f:
            json.dump(prov, f, indent=2, sort_keys=True)
        print(f"Generated Provenance at: {out_prov}")
    except Exception as e:
        sys.exit(f"Failed to write Provenance: {e}")


def cmd_check_toolchain(lockfile: str) -> None:
    """Validate a toolchain lockfile (stub).

    Args:
        lockfile: Path to the toolchain lockfile to validate.

    Note:
        This is a stub implementation and does not yet perform validation.
    """
    print("CheckToolchain not implemented yet")


# --- CLI ---


def main() -> None:
    """Run the minimal STUNIR CLI dispatcher.

    Parses command-line arguments and dispatches to the appropriate command handler.
    Supported commands: spec-to-ir, gen-provenance, check-toolchain.
    """
    parser = argparse.ArgumentParser(description="STUNIR Minimal Python Toolchain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # spec-to-ir
    p_spec = subparsers.add_parser("spec-to-ir")
    p_spec.add_argument("--in-json", required=True)
    p_spec.add_argument("--out-ir", required=True)

    # gen-provenance
    p_prov = subparsers.add_parser("gen-provenance")
    p_prov.add_argument("--in-ir", required=True)
    p_prov.add_argument("--epoch-json", required=True)
    p_prov.add_argument("--out-prov", required=True)

    # check-toolchain
    p_check = subparsers.add_parser("check-toolchain")
    p_check.add_argument("--lockfile", required=True)

    args = parser.parse_args()

    if args.command == "spec-to-ir":
        cmd_spec_to_ir(args.in_json, args.out_ir)
    elif args.command == "gen-provenance":
        cmd_gen_provenance(args.in_ir, args.epoch_json, args.out_prov)
    elif args.command == "check-toolchain":
        cmd_check_toolchain(args.lockfile)

if __name__ == "__main__":
    main()
