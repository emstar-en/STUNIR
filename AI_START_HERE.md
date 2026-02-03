# AI_START_HERE

## Ada SPARK is the DEFAULT Implementation Language

**STUNIR uses Ada SPARK as its DEFAULT implementation language.**

Python files are fully functional implementations suitable for many use cases:
- Use Ada SPARK tools (`tools/spark/`) for safety-critical, formally verified operations
- Use Python tools (`tools/*.py`) for rapid prototyping, development, and non-critical applications
- All build scripts default to Ada SPARK, but Python is available as an alternative

---

## Quick Navigation

**Canonical entrypoint:** [`ENTRYPOINT.md`](ENTRYPOINT.md)

### Primary Tools (Ada SPARK)
- **Spec to IR**: `tools/spark/bin/stunir_spec_to_ir_main`
- **IR to Code**: `tools/spark/bin/stunir_ir_to_code_main`
- **Build**: See `tools/spark/README.md` for building Ada SPARK tools

### Python Tools (Alternative Pipeline)
- `tools/spec_to_ir.py` - Fully functional spec-to-IR converter
- `tools/ir_to_code.py` - Fully functional IR-to-code emitter
- Use when: rapid prototyping, ease of modification, or when SPARK toolchain is unavailable

---

## When to Use Each Pipeline

| Use Case | Recommended Pipeline | Reason |
|----------|---------------------|--------|
| Safety-critical systems | Ada SPARK | Formal verification, DO-178C compliance |
| Reproducible builds | Ada SPARK | Deterministic execution |
| Rapid prototyping | Python | Faster iteration, easier debugging |
| Custom modifications | Python | Easier to read and modify |
| CI/CD pipelines | Either | SPARK for verification, Python for speed |
| Learning/experimentation | Python | More accessible, well-documented |

## Rationale

Ada SPARK was chosen as the default language because:
1. **Formal Verification**: SPARK proofs guarantee absence of runtime errors
2. **Determinism**: Predictable execution for reproducible builds
3. **Safety**: Strong typing prevents entire classes of bugs
4. **DO-178C Compliance**: Industry standard for safety-critical systems
5. **Performance**: Native compilation, no interpreter overhead

---

**If you are an agent/human trying to orient yourself in this repository, open [`ENTRYPOINT.md`](ENTRYPOINT.md).**
