# STUNIR Shell-Native Implementation (Profile 3)

The **Shell-Native** profile is a minimal, dependency-free implementation of the STUNIR core logic. It is designed for:
1.  **Bootstrapping:** Setting up the environment before Python/Native tools are available.
2.  **Verification:** Independently verifying receipts and checksums.
3.  **Restricted Environments:** Running in highly constrained CI/CD containers where Python is absent.

## Capabilities

| Feature | Status | Description |
| :--- | :--- | :--- |
| **Toolchain Discovery** | ✅ Complete | Scans for `git`, `python`, `bash`, hashes them, and generates `local_toolchain.lock.json`. |
| **Receipt Generation** | ✅ Complete | Generates `receipt.json` with Merkle roots of inputs, toolchain, and outputs. |
| **JSON Generation** | ✅ Complete | Dependency-free JSON writer (`scripts/lib/json.sh`). |
| **Hashing** | ✅ Complete | SHA-256 calculation using `sha256sum` or `shasum`. |
| **IR Generation** | 🚧 Stub | Currently simulates generation. Full IR compiler pending. |
| **Code Emission** | 🚧 Stub | Currently simulates emission. |

## Architecture

The shell implementation resides in `scripts/lib/`:

*   **`core.sh`**: Shared utilities (logging, error handling, hashing).
*   **`json.sh`**: A lightweight JSON builder.
*   **`manifest.sh`**: Toolchain discovery and lockfile generation.
*   **`receipt.sh`**: Receipt generation logic.
*   **`runner.sh`**: The main entry point/dispatcher for shell mode.

## Usage

The shell runner is automatically invoked by `scripts/build.sh` if Python is missing. You can also invoke it directly or force it:

```bash
# Force Shell Mode
STUNIR_PROFILE=shell ./scripts/build.sh build
```

## Limitations

*   **JSON Parsing:** The shell implementation uses simple `grep`/`sed` for JSON extraction. It is **not** a full JSON parser and expects formatted input (e.g., one key per line).
*   **Performance:** Shell scripts are significantly slower than Python or Native binaries for complex logic.
*   **Platform:** Requires a POSIX-compliant shell (Bash 4+ recommended) and standard coreutils.
