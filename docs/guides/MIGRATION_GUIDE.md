# STUNIR Migration Guide

## Version 0.8.9

This guide helps users migrate between STUNIR versions and understand breaking changes.

### Migrating from v0.8.8 to v0.8.9

#### Changes
- **Version strings standardized** to 0.8.9 across all components
- **SPARK Optimizer** now includes constant folding, dead code elimination, and unreachable code elimination
- **Test suite** expanded with 25 optimizer tests
- **Documentation** improved with comprehensive guides

#### No Breaking Changes
This is a patch release with no breaking changes. All v0.8.8 specs and IR are compatible with v0.8.9.

### Migrating from v0.8.x to v0.8.9

#### Tool Chain Updates
1. **spec_to_ir**: No changes to command-line interface
2. **ir_to_code**: No changes to command-line interface
3. **Optimizer**: New optimization passes enabled by default

#### Recommended Steps
1. Update your STUNIR installation
2. Rebuild any custom tools
3. Run your test suite
4. Verify output matches expectations

### Migrating from v0.7.x to v0.8.x

#### Major Changes
- **Ada SPARK** is now the default implementation
- **Python tools** are now reference implementations
- **IR format** v2 with flattened control flow
- **New emitters**: GPU, WASM, Assembly, Polyglot

#### Breaking Changes
1. **IR Format**: v2 is not backward compatible with v1
2. **Tool Paths**: Ada SPARK binaries in different location
3. **Build System**: Now uses gprbuild for Ada components

#### Migration Steps
1. **Install Ada SPARK toolchain**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install gnat gprbuild
   
   # macOS
   brew install gnat
   
   # Windows
   # Download from AdaCore or use Alire
   ```

2. **Update build scripts**:
   ```bash
   # Old (Python)
   python tools/spec_to_ir.py ...
   
   # New (Ada SPARK)
   tools/spark/bin/stunir_spec_to_ir_main ...
   # OR use precompiled binaries
   precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main ...
   ```

3. **Convert specs to new format**:
   ```bash
   # If you have old v1 specs, convert them
   python tools/migrate_spec.py --from v1 --to v2 old_spec.json new_spec.json
   ```

4. **Update CI/CD pipelines**:
   - Replace Python tool calls with Ada SPARK binaries
   - Add gprbuild to build environment
   - Update path references

### Common Migration Issues

#### Issue: "gprbuild: command not found"
**Solution**: Install GNAT toolchain with gprbuild

#### Issue: "spec_to_ir: module not found"
**Solution**: Use Ada SPARK binary instead of Python module

#### Issue: "IR version mismatch"
**Solution**: Regenerate IR from specs using new tools

### Version Compatibility Matrix

| STUNIR Version | IR Version | Python Tools | Ada SPARK Tools | Status |
|----------------|------------|--------------|-----------------|--------|
| 0.9.0 (future) | v3 | Deprecated | Primary | Planned |
| 0.8.9 | v2 | Reference | Primary | Current |
| 0.8.0-0.8.8 | v2 | Reference | Primary | Supported |
| 0.7.x | v1 | Primary | N/A | Deprecated |
| 0.6.x | v1 | Primary | N/A | Deprecated |

### Deprecation Timeline

- **Python tools**: Will be deprecated in v0.9.0, removed in v1.0.0
- **IR v1**: No longer supported as of v0.8.0
- **Old CLI**: Migrated to new unified CLI in v0.8.0

### Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: File at project issue tracker
- **Discussions**: Join community forums

---

Last updated: 2026-02-03