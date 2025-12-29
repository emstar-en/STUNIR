# STUNIR Anti-Patterns & Forbidden Actions

## 🛑 CRITICAL: READ BEFORE GENERATING CODE

STUNIR is a **deterministic build system**. Standard CI/CD practices often violate its core theorems.
If you suggest any of the following, you are breaking the build.

### 1. The "PATH" Trap
- **BAD:** `python3 script.py` or `gcc main.c` (Relying on system PATH)
- **GOOD:** `$STUNIR_TOOL_PYTHON script.py` or `$STUNIR_TOOL_CC main.c`
- **WHY:** `PATH` is a determinism leak. We do not know which version of a tool we are getting.

### 2. The "Network" Trap
- **BAD:** `pip install requests`, `npm install`, `cargo build` (fetching crates)
- **GOOD:** All dependencies must be vendored in `inputs/` or `tools/`.
- **WHY:** The build phase is **OFFLINE**. Network access is only allowed in the explicit "Discovery/Fetch" phase, never in `build.sh`.

### 3. The "Timestamp" Trap
- **BAD:** `datetime.now()`, `date`, `touch file`
- **GOOD:** Use `STUNIR_BUILD_EPOCH` or `SOURCE_DATE_EPOCH`.
- **WHY:** Builds must be bit-for-bit identical whether run today or next year.

### 4. The "JSON" Trap
- **BAD:** `json.dump(data, f)` (Default formatting)
- **GOOD:** `json.dump(data, f, sort_keys=True, separators=(',', ':'))`
- **WHY:** JSON serialization is non-deterministic by default (whitespace, key order). We need Canonical JSON for hashing.

### 5. The "Python Dependency" Trap
- **BAD:** Assuming Python is available for orchestration logic.
- **GOOD:** Write orchestration in POSIX `sh`. Use Python only if `STUNIR_TOOL_PYTHON` is set.
- **WHY:** STUNIR must run in "Shell Primary" mode (Profile 3) on constrained systems.

### 6. The "AbsolutePath" Trap (in Artifacts)
- **BAD:** Embedding `/home/user/stunir/build/...` in a binary or generated file.
- **GOOD:** Use relative paths or deterministic placeholders (`/src`, `/build`).
- **WHY:** The build directory varies per user. Artifacts must be portable.
