# STUNIR Anti-Patterns

To maintain determinism and verification standards, avoid the following patterns in the STUNIR codebase.

## 1. Implicit Toolchains
**Bad:** Relying on `git`, `python`, or `gcc` being in the system `PATH` without verification.
**Good:** Use the `local_toolchain.lock.json` to resolve absolute paths and verify hashes before execution.

## 2. Non-Deterministic Output
**Bad:** Using `datetime.now()` or random seeds in generated code or IR.
**Good:** All outputs must be a pure function of the Input Spec. Timestamps in receipts are allowed but must be clearly separated from the semantic content.

## 3. Hardcoded Paths
**Bad:** `C:\Windows\System32` or `/usr/bin/python`.
**Good:** Discover paths dynamically during the `discover` phase and lock them.

## 4. Shell Dependency Creep
**Bad:** Using `jq`, `curl`, or `perl` in core shell scripts.
**Good:** Stick to POSIX standard utilities (`sed`, `awk`, `grep`, `cat`) for the Shell-Native profile to ensure maximum portability.

## 5. "Works on My Machine"
**Bad:** Assuming a specific environment (e.g., "It works because I have Python 3.11 installed").
**Good:** The Polyglot Build System (`scripts/build.sh`) must handle environment detection and graceful degradation (e.g., falling back to Shell mode).
