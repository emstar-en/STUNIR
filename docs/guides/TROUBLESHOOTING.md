# STUNIR Troubleshooting Guide

## Version 0.8.9

Common issues and their solutions when using STUNIR.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Build Issues](#build-issues)
3. [Runtime Issues](#runtime-issues)
4. [Test Failures](#test-failures)
5. [Platform-Specific Issues](#platform-specific-issues)

---

## Installation Issues

### Issue: "gprbuild: command not found"

**Symptom:**
```
gprbuild: command not found
```

**Cause:** GNAT toolchain not installed or not in PATH.

**Solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install gnat gprbuild
```

**macOS:**
```bash
brew install gnat
```

**Windows:**
1. Download from [AdaCore](https://www.adacore.com/download) or use [Alire](https://alire.ada.dev/)
2. Add to PATH: `C:\gnat\bin` or `C:\Users\<user>\.alire\libexec\spark\bin`

**Verify:**
```bash
gprbuild --version
```

---

### Issue: "cargo: command not found"

**Symptom:**
```
cargo: command not found
```

**Cause:** Rust toolchain not installed.

**Solution:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Verify:**
```bash
rustc --version
cargo --version
```

---

### Issue: "python: module not found"

**Symptom:**
```
ModuleNotFoundError: No module named 'stunir'
```

**Cause:** Python package not installed or wrong Python version.

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

**Verify:**
```bash
python -c "import stunir; print(stunir.__version__)"
```

---

## Build Issues

### Issue: "project file not found"

**Symptom:**
```
gprbuild: project file "stunir_tools.gpr" not found
```

**Cause:** Running gprbuild from wrong directory.

**Solution:**
```bash
cd tools/spark
gprbuild -P stunir_tools.gpr
```

---

### Issue: "compilation errors in Ada code"

**Symptom:**
```
error: invalid syntax
error: undefined symbol
```

**Cause:** Syntax errors or missing dependencies.

**Solution:**
1. Check Ada version: `gnat --version`
2. Ensure minimum version: GNAT 12.0+
3. Clean build: `gprclean -P stunir_tools.gpr && gprbuild -P stunir_tools.gpr`

---

### Issue: "Rust linking errors on Windows"

**Symptom:**
```
error: linking with `gcc` failed
undefined reference to `__imp___acrt_iob_func`
```

**Cause:** MinGW/MSYS2 toolchain incompatibility.

**Solution:**

**Option 1: Use MSVC toolchain**
```bash
rustup default stable-x86_64-pc-windows-msvc
```

**Option 2: Use GNU toolchain with correct MinGW**
```bash
rustup default stable-x86_64-pc-windows-gnu
# Ensure MinGW is in PATH before Windows system directories
```

**Option 3: Use WSL2**
```bash
# In WSL2 Ubuntu
sudo apt-get install build-essential
cargo build
```

---

### Issue: "gprbuild times out in terminal"

**Symptom:**
```
[Command timed out]
```

**Cause:** gprbuild takes longer than terminal timeout limit.

**Solution:**
1. Run build manually from correct directory
2. Use background process: `gprbuild -P stunir_tools.gpr &`
3. Increase terminal timeout if configurable

---

## Runtime Issues

### Issue: "IR version mismatch"

**Symptom:**
```
Error: IR version 1.0, expected 2.0
```

**Cause:** Using old IR with new tools.

**Solution:**
```bash
# Regenerate IR from specs
stunir_spec_to_ir_main --spec-root specs/ --out new.ir.json
```

---

### Issue: "spec_to_ir: no JSON specs found"

**Symptom:**
```
FileNotFoundError: No JSON specs found in specs/
```

**Cause:** Directory empty or wrong path.

**Solution:**
```bash
# Verify directory contents
ls specs/

# Check file extensions (must be .json)
ls specs/*.json
```

---

### Issue: "ir_to_code: target not supported"

**Symptom:**
```
Error: Target language 'java' not supported
```

**Cause:** Requested target not implemented.

**Solution:**
Use supported targets: `rust`, `c`, `python`, `js`, `zig`, `go`, `ada`

---

### Issue: "optimizer: pass not found"

**Symptom:**
```
Error: Optimization pass 'inline-functions' not found
```

**Cause:** Requested pass doesn't exist.

**Solution:**
Use available passes:
- `constant-folding`
- `constant-propagation`
- `dead-code-elimination`
- `unreachable-code-elimination`

---

## Test Failures

### Issue: "Rust tests fail to link"

**Symptom:**
```
error: could not compile `stunir`
caused by: linking with `gcc` failed
```

**Cause:** Windows toolchain configuration.

**Solution:**
See [Rust linking errors on Windows](#issue-rust-linking-errors-on-windows)

---

### Issue: "SPARK tests: field not found"

**Symptom:**
```
error: no selector "Type_Param_Cnt" for type "IR_Function"
```

**Cause:** Test code using outdated field names.

**Solution:**
Fixed in v0.8.9. Update to latest version.

---

### Issue: "Python tests: assertion error"

**Symptom:**
```
AssertionError: Expected 5, got 4
```

**Cause:** Test expectations don't match implementation.

**Solution:**
1. Check test is up to date
2. Verify implementation behavior
3. Update test or fix implementation

---

## Platform-Specific Issues

### Windows

#### Issue: "Path too long"

**Symptom:**
```
error: path too long
```

**Cause:** Windows MAX_PATH limitation (260 characters).

**Solution:**
```powershell
# Enable long path support (Windows 10 1607+)
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use short paths
subst X: C:\Very\Long\Path\To\Project
cd X:\
```

---

#### Issue: "Permission denied"

**Symptom:**
```
PermissionError: [WinError 5] Access is denied
```

**Cause:** Insufficient permissions or file in use.

**Solution:**
```powershell
# Run as Administrator
# Or close applications using the files
# Or check antivirus software
```

---

### macOS

#### Issue: "gnat not found after brew install"

**Symptom:**
```
gnat: command not found
```

**Cause:** PATH not updated.

**Solution:**
```bash
# Add to ~/.zshrc or ~/.bash_profile
export PATH="/opt/homebrew/opt/gnat/bin:$PATH"

# Reload shell
source ~/.zshrc
```

---

### Linux

#### Issue: "gprbuild: cannot find gnat1"

**Symptom:**
```
gprbuild: cannot find gnat1
```

**Cause:** GNAT installation incomplete.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install gnat-12 gprbuild

# Or install full GNAT package
sudo apt-get install gnat
```

---

## Getting More Help

### Enable Debug Logging

**Python:**
```bash
export STUNIR_DEBUG=1
python tools/spec_to_ir.py ...
```

**Ada SPARK:**
```bash
stunir_spec_to_ir_main --verbose ...
```

### Check System Information

```bash
# OS
uname -a

# GNAT
gnat --version
gprbuild --version

# Rust
rustc --version
cargo --version

# Python
python --version
pip list | grep stunir
```

### Report Issues

When reporting issues, include:
1. STUNIR version: `cat VERSION`
2. Operating system and version
3. Tool versions (GNAT, Rust, Python)
4. Complete error message
5. Steps to reproduce
6. Expected vs actual behavior

---

Last updated: 2026-02-03