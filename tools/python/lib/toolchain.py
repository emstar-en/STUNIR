import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

class ToolchainError(Exception):
    pass

class Toolchain:
    def __init__(self, lockfile_path: str = "local_toolchain.lock.json"):
        self.lockfile_path = Path(lockfile_path).resolve()
        self.tools: Dict[str, Dict] = {}
        self._loaded = False

    def load(self, strict: bool = True):
        if not self.lockfile_path.exists():
            if strict:
                raise ToolchainError(f"Lockfile not found at {self.lockfile_path}. Run discovery first.")
            return

        try:
            with open(self.lockfile_path, "r") as f:
                data = json.load(f)

            # Handle both list (v1.0) and dict formats if schema evolves
            # Current shell script outputs: { "tools": [ ... ] }
            if "tools" in data and isinstance(data["tools"], list):
                for t in data["tools"]:
                    self.tools[t["name"]] = t
            else:
                # Fallback or alternative schema
                pass

            self._loaded = True

        except json.JSONDecodeError as e:
            raise ToolchainError(f"Invalid lockfile JSON: {e}")

    def verify_tool(self, name: str) -> str:
        """
        Verifies a tool's hash matches the lockfile.
        Returns the absolute path to the tool.
        """
        if not self._loaded:
            self.load()

        if name not in self.tools:
            raise ToolchainError(f"Tool '{name}' not found in lockfile.")

        tool_info = self.tools[name]
        path = tool_info.get("path")
        expected_hash = tool_info.get("sha256")
        status = tool_info.get("status")

        if status != "OK" or not path:
            raise ToolchainError(f"Tool '{name}' is marked as {status} or missing path.")

        # Verify Hash
        current_hash = self._hash_file(path)
        if current_hash != expected_hash:
            raise ToolchainError(
                f"Tool '{name}' hash mismatch!\n"
                f"  Expected: {expected_hash}\n"
                f"  Found:    {current_hash}\n"
                f"  Path:     {path}"
            )

        return path

    def get_tool(self, name: str) -> Optional[str]:
        """
        Safe getter. Returns path if verified, None otherwise.
        """
        try:
            return self.verify_tool(name)
        except ToolchainError:
            return None

    def _hash_file(self, path: str) -> str:
        sha = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk: break
                    sha.update(chunk)
            return sha.hexdigest()
        except OSError:
            return "MISSING"

# Singleton instance
_instance = Toolchain()

# Module-level exports
def require(name: str) -> str:
    return _instance.verify_tool(name)

def get_tool(name: str) -> Optional[str]:
    return _instance.get_tool(name)

def load(path: str = "local_toolchain.lock.json"):
    _instance.lockfile_path = Path(path).resolve()
    _instance.load()
