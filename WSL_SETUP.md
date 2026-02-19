# STUNIR WSL Setup Guide

## Migration Complete ✓

Your STUNIR project has been successfully migrated to WSL (Windows Subsystem for Linux).

## Location

- **Windows Path**: `C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\STUNIR`
- **WSL Path**: `/home/mstar/STUNIR`

## Quick Start

### 1. Access your WSL STUNIR directory

```bash
wsl
cd /home/mstar/STUNIR
```

### 2. Run the setup script

```bash
./setup_wsl.sh
```

This will:
- Install required dependencies (Python3, pip, git, build tools)
- Check for optional tools (Rust, Haskell/GHC, GNAT/Ada)
- Install Python packages
- Run toolchain discovery
- Make all scripts executable

### 3. Test the installation

```bash
./test_pipeline.sh
```

## Key Files Created/Converted

- `setup_wsl.sh` - WSL environment bootstrap script
- `test_pipeline.sh` - Bash version of test_pipeline.ps1

## Accessing from VSCode

You can open the WSL directory directly in VSCode:

1. **From VSCode**: Use the Remote-WSL extension
   - Open Command Palette (Ctrl+Shift+P)
   - Select "Remote-WSL: New Window"
   - Navigate to `/home/mstar/STUNIR`

2. **From Windows Terminal/PowerShell**:
   ```powershell
   wsl code /home/mstar/STUNIR
   ```

3. **From WSL Terminal**:
   ```bash
   cd /home/mstar/STUNIR
   code .
   ```

## Working Between Windows and WSL

- **Access Windows files from WSL**: `/mnt/c/Users/MSTAR/...`
- **Access WSL files from Windows**: `\\wsl$\Ubuntu\home\mstar\STUNIR`

## Next Steps

1. Run the setup script: `./setup_wsl.sh`
2. Install optional toolchains as needed (Rust, Haskell, GNAT)
3. Run tests: `./test_pipeline.sh`
4. Build tools: `bash scripts/build.sh`
5. Explore the project: `ls examples/`

## Optional Dependencies

### Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Install Haskell (GHC)
```bash
sudo apt-get update
sudo apt-get install ghc cabal-install
```

### Install GNAT (Ada)
```bash
sudo apt-get update
sudo apt-get install gnat gprbuild
```

## Troubleshooting

### Permission Issues
If you encounter permission issues, make scripts executable:
```bash
chmod +x setup_wsl.sh test_pipeline.sh
find scripts -name "*.sh" -type f -exec chmod +x {} \;
```

### Python Package Issues
If pip installation fails:
```bash
sudo apt-get install python3-pip python3-dev
pip3 install --upgrade pip
pip3 install -e . --user
```

### Line Ending Issues
If you see `\r` errors, convert line endings:
```bash
sudo apt-get install dos2unix
dos2unix setup_wsl.sh test_pipeline.sh
find scripts -name "*.sh" -type f -exec dos2unix {} \;
```

## Project Structure

The STUNIR directory contains:
- `precompiled/linux-x86_64/` - Precompiled binaries for Linux
- `scripts/` - Build and utility scripts
- `tools/` - Python implementation tools
- `src/` - Source code
- `examples/` - Example inputs and specs
- `tests/` - Test suite

For more information, see the main [README.md](README.md)
