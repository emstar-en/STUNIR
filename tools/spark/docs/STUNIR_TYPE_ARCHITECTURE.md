> **SUPERSEDED** — This document has been superseded by `tools/spark/ARCHITECTURE.md`
> (Section 4: Type System). The analysis here is preserved for historical reference.
> When this document and `ARCHITECTURE.md` conflict, `ARCHITECTURE.md` is correct.

---

# STUNIR Type System Architecture

## Overview

The STUNIR project maintains two parallel implementations of the type system to balance formal verification requirements with practical development needs.

## Architecture Decision

### Two-Tier Type System

1. **Main SPARK Implementation** (`src/stunir_types.ads`, `src/stunir_json_parser.ads`)
   - **Purpose**: Formal verification and safety-critical components
   - **Mode**: `pragma SPARK_Mode (On)`
   - **Features**:
     - Full formal contracts and preconditions
     - Bounded strings for all data (provably safe)
     - Comprehensive status codes (19 error types)
     - Complex parser state with nesting tracking
     - Suitable for DO-333 certification

2. **Powertools Implementation** (`src/powertools/stunir_types.ads`, `src/powertools/stunir_json_parser.ads`)
   - **Purpose**: Rapid CLI tool development
   - **Mode**: `pragma SPARK_Mode (Off)`
   - **Features**:
     - Simplified API without formal contracts
     - Mix of bounded/unbounded strings for flexibility
     - Minimal status codes (3 types)
     - Simple parser state
     - Pragmatic error handling

## Type Compatibility

### Token Types
- **Main SPARK**: `Token_Kind` type
- **Powertools**: `Token_Type` type
- **Compatibility**: Both define identical tokens with compatible aliases

### String Types
- **Main SPARK**: `JSON_String` (Bounded_String, 1MB max)
- **Powertools**: `JSON_String` (Bounded_String, 1MB max) + `Unbounded_String` for flexibility
- **Compatibility**: JSON_String is identical, conversion functions available

### Parser State
- **Main SPARK**: Complex with nesting stack, formal contracts
- **Powertools**: Simplified with Position/Line/Column tracking
- **Compatibility**: Core fields (Input, Position, Current_Token) are compatible

### Status Codes
- **Main SPARK**: `Status_Code` with 19 specific error types
- **Powertools**: `Status_Code` with 3 general types (Success, Error, EOF_Reached)
- **Compatibility**: Main SPARK errors can map to powertools' Error type

## Usage Guidelines

### When to Use Main SPARK Types
- Safety-critical code paths
- Code requiring formal verification
- Components that need DO-333 certification
- Low-level parsing and validation

### When to Use Powertools Types
- CLI utilities and development tools
- Prototype and experimental features
- Code that needs dynamic string handling
- Rapid development without certification requirements

## Integration Points

### Conversion Functions (Recommended)
```ada
function To_Powertools_Status (SPARK_Status : STUNIR_Types.Status_Code) 
   return Stunir_Types.Status_Code;

function From_Bounded_String (Input : STUNIR_Types.JSON_String) 
   return Unbounded_String;
```

### Current Status
- Both type systems are documented and stable
- Powertools can coexist with main SPARK implementation
- No automatic conversion layer currently implemented
- Each subsystem is self-contained

## Design Rationale

This dual-type-system approach allows:
1. **Formal Verification**: Core STUNIR components can achieve formal proofs
2. **Rapid Development**: Powertools can evolve quickly without proof obligations
3. **Clear Boundaries**: Each system has well-defined scope and usage
4. **Future Flexibility**: Conversion layers can be added as needed

## Maintenance Notes

- Keep type definitions synchronized for JSON_String size and token types
- Document any breaking changes in both implementations
- Consider adding conversion utilities if cross-subsystem communication increases
- Review annually whether integration layer is needed

---
**Version**: 1.0  
**Last Updated**: 2026-02-17  
**Author**: STUNIR Development Team
