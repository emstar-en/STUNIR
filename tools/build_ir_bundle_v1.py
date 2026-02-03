"""CLI entry point for building STUNIR IR bundle v1.

Kept as a tiny shim so callers can treat it like a tool script.
"""

from tools.ir_bundle_v1 import _main


if __name__ == "__main__":
    raise SystemExit(_main())
