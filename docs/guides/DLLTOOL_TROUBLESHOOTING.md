# dlltool.exe Troubleshooting Guide

**Issue:** Rust tests fail with `error calling dlltool 'dlltool.exe': program not found`

**Root Cause:** dlltool.exe is part of MinGW/binutils but may not be in PATH after installation.

---

## Quick Fix (Option 1): Use MSYS2 MinGW

Since MSYS2 was installed, you need to install MinGW within it:

### Step 1: Open MSYS2 Terminal

1. Press `Win + R`
2. Type `msys2` and press Enter
3. Or search for "MSYS2" in Start Menu

### Step 2: Install MinGW-w64

In the MSYS2 terminal, run:

```bash
# Update package database
pacman -Syu

# Install MinGW-w64 toolchain (includes dlltool)
pacman -S mingw-w64-x86_64-toolchain

# When prompted, select 'all' or press Enter to accept defaults
```

### Step 3: Add to Windows PATH

After installation, add MinGW to your Windows PATH:

```powershell
# In PowerShell (as Administrator)
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\msys64\mingw64\bin",
    "Machine"
)
```

Or manually:
1. Open System Properties → Advanced → Environment Variables
2. Edit "Path" variable
3. Add `C:\msys64\mingw64\bin`

### Step 4: Verify Installation

```powershell
# In new PowerShell window
where.exe dlltool.exe
# Expected: C:\msys64\mingw64\bin\dlltool.exe

dlltool --version
# Expected: GNU dlltool (GNU Binutils) 2.x.x
```

---

## Alternative Fix (Option 2): Use LLVM/Clang

If MinGW continues to cause issues, you can use LLVM's dlltool:

```powershell
# Install LLVM (includes llvm-dlltool)
winget install --id=LLVM.LLVM -e

# Add to PATH
$env:PATH += ";C:\Program Files\LLVM\bin"

# Create symlink for compatibility
New-Item -ItemType SymbolicLink -Path "C:\Program Files\LLVM\bin\dlltool.exe" -Target "C:\Program Files\LLVM\bin\llvm-dlltool.exe"
```

---

## Alternative Fix (Option 3): Use Windows Subsystem for Linux (WSL)

If Windows native tools continue to fail, use WSL:

```powershell
# Install WSL
wsl --install

# After restart, in WSL terminal:
sudo apt-get update
sudo apt-get install -y mingw-w64 gcc-mingw-w64-x86-64

# Run Rust tests in WSL
cd /mnt/c/Users/MSTAR/AppData/Roaming/AbacusAI/Agent\ Workspaces/STUNIR-main/tools/rust
cargo test --lib
```

---

## Alternative Fix (Option 4): Skip Windows Tests (Use CI)

If local Windows setup is problematic, rely on GitHub Actions for Windows testing:

1. Push code to GitHub
2. GitHub Actions will run tests on Windows runner with proper MinGW setup
3. Check CI results before merging

This is documented in the recovery plan's Windows CI configuration.

---

## Verification Commands

After any fix, verify with:

```powershell
# Check dlltool location
Get-Command dlltool.exe

# Check version
dlltool --version

# Test Rust compilation
cd STUNIR-main/tools/rust
cargo build

# Run tests
cargo test --lib
```

---

## Common Issues

### Issue: "pacman is not recognized"
**Fix:** Make sure you're running the MSYS2 terminal, not Windows PowerShell

### Issue: "Permission denied" when adding to PATH
**Fix:** Run PowerShell as Administrator

### Issue: "dlltool.exe found but cargo still fails"
**Fix:** Restart PowerShell/terminal to reload PATH

### Issue: "incompatible architecture" errors
**Fix:** Make sure you installed x86_64 version, not i686:
```bash
# In MSYS2, install x86_64 specifically:
pacman -S mingw-w64-x86_64-binutils
```

---

## Summary

| Method | Difficulty | Reliability | Recommendation |
|--------|------------|-------------|----------------|
| MSYS2 MinGW | Medium | High | **Recommended** |
| LLVM | Easy | Medium | Good alternative |
| WSL | Easy | High | Best for Linux users |
| CI Only | Trivial | High | If local setup fails |

---

**Next Steps:**
1. Try Option 1 (MSYS2 MinGW) first
2. If issues persist, try Option 2 (LLVM)
3. As last resort, use WSL or CI-only testing

**Related:** See `V0.9_RECOVERY_PLAN.md` Phase 4 for complete Rust test setup instructions.