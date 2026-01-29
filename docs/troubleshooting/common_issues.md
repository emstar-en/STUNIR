# STUNIR Common Issues

> Part of `docs/troubleshooting/1070`

## Build Issues

### Issue: ModuleNotFoundError: No module named 'tools'

**Symptom:**
```
ModuleNotFoundError: No module named 'tools'
```

**Cause:** Python path not set correctly.

**Solution:**
```bash
# Option 1: Run from repository root
cd /path/to/STUNIR
python -m tools.ir_emitter.emit_ir spec.json

# Option 2: Set PYTHONPATH
export PYTHONPATH=/path/to/STUNIR:$PYTHONPATH

# Option 3: Install as package
pip install -e .
```

---

### Issue: Permission denied: scripts/build.sh

**Symptom:**
```
bash: ./scripts/build.sh: Permission denied
```

**Solution:**
```bash
chmod +x scripts/*.sh
chmod +x scripts/lib/*.sh
```

---

### Issue: stunir-native: command not found

**Symptom:**
```
stunir-native: command not found
```

**Cause:** Haskell native tools not built or not in PATH.

**Solutions:**

```bash
# Option 1: Build native tools
cd tools/native/haskell/stunir-native
cabal build
cabal install --installdir=$HOME/.local/bin
export PATH=$HOME/.local/bin:$PATH

# Option 2: Use Python profile (no native tools needed)
./scripts/build.sh --profile=python
```

---

## Verification Issues

### Issue: Hash mismatch in manifest

**Symptom:**
```
[ERROR] Hash mismatch for asm/ir/module.dcbor
Expected: sha256:abc123...
Actual: sha256:def456...
```

**Cause:** Artifact was modified after manifest generation.

**Solution:**
```bash
# Regenerate manifests
python -m manifests.ir.gen_ir_manifest --output receipts/ir_manifest.json

# Or rebuild from scratch
rm -rf asm/ir/ receipts/
./scripts/build.sh
```

---

### Issue: Missing file in manifest

**Symptom:**
```
[ERROR] File not found: asm/ir/module.dcbor
```

**Cause:** Incomplete build or deleted file.

**Solution:**
```bash
# Check what's missing
ls -la asm/ir/

# Regenerate IR
python -m tools.ir_emitter.emit_ir spec.json asm/ir/module.json

# Or full rebuild
./scripts/build.sh
```

---

### Issue: Manifest not found

**Symptom:**
```
[ERROR] receipts/ir_manifest.json not found
```

**Solution:**
```bash
# Generate manifests
mkdir -p receipts
python -m manifests.ir.gen_ir_manifest
python -m manifests.receipts.gen_receipts_manifest
```

---

## Determinism Issues

### Issue: Different output on each build

**Symptom:** SHA-256 hashes differ between builds.

**Diagnosis:**
```bash
# Build twice and compare
./scripts/build.sh
cp receipts/ir_manifest.json /tmp/manifest1.json

./scripts/build.sh
cp receipts/ir_manifest.json /tmp/manifest2.json

diff /tmp/manifest1.json /tmp/manifest2.json
```

**Common Causes:**

1. **Timestamps**: Using `datetime.now()` instead of epochs
2. **Dict ordering**: Not using `sort_keys=True`
3. **File ordering**: Not sorting file lists
4. **Float precision**: Different float representations

**Solutions:**

```python
# Use canonical JSON
import json
json.dumps(data, sort_keys=True, separators=(',', ':'))

# Use epochs
epoch = int(os.environ.get('STUNIR_EPOCH', '1735500000'))

# Sort file lists
files = sorted(os.listdir(directory))
```

---

### Issue: epoch field changes

**Symptom:** Epoch differs between builds.

**Solution:**
```bash
# Set fixed epoch
export STUNIR_EPOCH=1735500000
./scripts/build.sh
```

---

## Profile Issues

### Issue: Native profile fails

**Symptom:**
```
[ERROR] Haskell toolchain not found
```

**Solution:**
```bash
# Install Haskell toolchain
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
ghcup install ghc 9.4.7
ghcup install cabal 3.8.1.0

# Or use Python profile
./scripts/build.sh --profile=python
```

---

### Issue: Shell profile missing jq

**Symptom:**
```
jq: command not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install jq

# macOS
brew install jq

# RHEL/CentOS
sudo yum install jq
```

---

## Performance Issues

### Issue: Build is slow

**Possible causes:**
1. Large spec files
2. Many targets enabled
3. Disk I/O bottleneck

**Solutions:**
```bash
# Use native profile for speed
./scripts/build.sh --profile=native

# Build specific target only
python -m tools.emitters.emit_code --target rust --ir ir.json

# Use SSD storage
```

---

## Still Having Issues?

1. Check logs: `cat build.log`
2. Enable debug: `STUNIR_DEBUG=1 ./scripts/build.sh`
3. Check disk space: `df -h`
4. Check memory: `free -h`
5. Open an issue with diagnostic output

## Related
- [Troubleshooting Overview](README.md)
- [Deployment Guide](../deployment/README.md)

---
*STUNIR Common Issues v1.0*
