#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import platform
from datetime import datetime, timezone
from typing import Dict, Optional, Any

# Import the core logic from minimal (Code Reuse)
# In a real package structure, this would be `from . import stunir_minimal`
# For now, we assume they are in the same dir or we import by path.
try:
    import stunir_minimal
except ImportError:
    # Fallback for standalone execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import stunir_minimal

class StunirFactory:
    def __init__(self, workspace_root: str):
        self.root = workspace_root
        self.build_dir = os.path.join(self.root, "build")
        self.lockfile = os.path.join(self.build_dir, "local_toolchain.lock.json")
        os.makedirs(self.build_dir, exist_ok=True)

    def log(self, msg: str):
        print(f"[STUNIR:FACTORY] {msg}")

    def discover_toolchain(self) -> Dict[str, Any]:
        self.log("Running Toolchain Discovery (Python-Railed)...")

        tools = {}

        # 1. Python (Self)
        tools["python"] = {"path": sys.executable, "required": True}

        # 2. Git
        git_path = subprocess.check_output(["which", "git"]).decode().strip()
        tools["git"] = {"path": git_path, "required": True}

        # 3. Bash (for shell fallbacks)
        bash_path = subprocess.check_output(["which", "bash"]).decode().strip()
        tools["bash"] = {"path": bash_path, "required": True}

        lock_data = {
            "schema": "v1",
            "tools": tools,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        with open(self.lockfile, 'w') as f:
            json.dump(lock_data, f, indent=2)

        self.log(f"Toolchain locked at {self.lockfile}")
        return lock_data

    def run_pipeline(self, spec_path: str):
        self.log(f"Starting Factory Pipeline for {spec_path}")

        # 1. Discovery
        if not os.path.exists(self.lockfile):
            self.discover_toolchain()

        # 2. Spec -> IR
        ir_path = os.path.join(self.build_dir, "factory.ir")
        self.log("Step 1: Spec -> IR")
        stunir_minimal.cmd_spec_to_ir(spec_path, ir_path)

        # 3. Epoch Generation (Factory Feature: Auto-Epoch)
        epoch_path = os.path.join(self.build_dir, "epoch.json")
        epoch_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "builder": f"stunir-factory-py-{platform.system()}",
            "mode": "factory"
        }
        with open(epoch_path, 'w') as f:
            json.dump(epoch_data, f, indent=2)

        # 4. Provenance
        prov_path = os.path.join(self.build_dir, "factory.prov")
        self.log("Step 2: Provenance")
        stunir_minimal.cmd_gen_provenance(ir_path, epoch_path, prov_path)

        self.log("Factory Pipeline Complete.")
        self.log(f"Artifacts: {ir_path}, {prov_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: stunir_factory.py <spec_file>")
        sys.exit(1)

    factory = StunirFactory(os.getcwd())
    factory.run_pipeline(sys.argv[1])

if __name__ == "__main__":
    main()
