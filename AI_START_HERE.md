# AI_START_HERE

## CRITICAL: Ada SPARK is the PRIMARY Implementation Language

**STUNIR uses Ada SPARK as its DEFAULT and PRIMARY implementation language.**

Python files exist only as reference implementations for readability. For all operations:
- Use Ada SPARK tools (`tools/spark/`) as the primary implementation
- Python tools (`tools/*.py`) are fallback/reference only
- All build scripts default to Ada SPARK

---

## Quick Navigation

**Canonical entrypoint:** [`ENTRYPOINT.md`](ENTRYPOINT.md)

### Primary Tools (Ada SPARK)
- **Spec to IR**: `tools/spark/bin/stunir_spec_to_ir_main`
- **IR to Code**: `tools/spark/bin/stunir_ir_to_code_main`
- **Build**: See `tools/spark/README.md` for building Ada SPARK tools

### Reference Implementations (Python - NOT for production)
- `tools/spec_to_ir.py` - Reference only
- `tools/ir_to_code.py` - Reference only

---

## Rationale

Ada SPARK was chosen as the primary language because:
1. **Formal Verification**: SPARK proofs guarantee absence of runtime errors
2. **Determinism**: Predictable execution for reproducible builds  
3. **Safety**: Strong typing prevents entire classes of bugs
4. **DO-178C Compliance**: Industry standard for safety-critical systems
5. **Performance**: Native compilation, no interpreter overhead

---

**If you are an agent/human trying to orient yourself in this repository, open [`ENTRYPOINT.md`](ENTRYPOINT.md).**
