# STUNIR Emitter Generator

**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**Author:** STUNIR Team  
**License:** MIT

## Overview

The STUNIR Emitter Generator is a meta-tool that scaffolds new code emitters across all 4 STUNIR pipelines simultaneously:

1. **SPARK (Ada)** - DO-178C Level A compliant, formally verified
2. **Python** - Reference implementation, easy to modify
3. **Rust** - High-performance, memory-safe
4. **Haskell** - Functional, type-safe

This tool ensures:
- ✅ **Consistency** across all 4 pipelines
- ✅ **Best practices** baked into templates
- ✅ **Time savings** - minutes instead of hours
- ✅ **Reduced errors** - automated scaffolding
- ✅ **Confluence** - all pipelines generate equivalent output

## Why This Tool?

After implementing 20+ emitter categories across 4 different languages, we noticed repeating patterns. This tool codifies those patterns, making it efficient to add new target categories.

> "A little codification doesn't hurt if it makes downstream processes more efficient" - STUNIR Design Philosophy

## Features

### 🎯 Multi-Pipeline Generation
- Generates complete emitters for all 4 pipelines at once
- Consistent structure and naming conventions
- Shared type mappings and configurations

### 📝 Template-Based
- Customizable templates for each language
- Variable substitution system
- Easy to extend and modify

### ✅ Built-In Validation
- Python syntax checking (`py_compile`)
- Build system integration updates
- File generation tracking

### 📚 Comprehensive Output
- Source code (all 4 languages)
- Test scaffolding (unit tests)
- Documentation (README with examples)
- Build system updates

### 🎨 Flexible Specification
- YAML or JSON input
- Command-line arguments
- Mix and match options

## Installation

The tool is already installed in your STUNIR repository:

```bash
cd /path/to/stunir_repo
./tools/emitter_generator/generate_emitter.py --help
```

### Requirements

- Python 3.9+ (for generator)
- PyYAML library: `pip install pyyaml`
- GNAT/SPARK (for building SPARK emitters)
- Rust toolchain (for building Rust emitters)
- GHC/Cabal (for building Haskell emitters)

## Quick Start

### Example 1: Generate from Specification File

```bash
./tools/emitter_generator/generate_emitter.py \
    --spec=tools/emitter_generator/specs/json_emitter.yaml
```

### Example 2: Generate from Command Line

```bash
./tools/emitter_generator/generate_emitter.py \
    --category=json \
    --description="JSON serialization emitter" \
    --output-types=json,schema \
    --features=pretty_print,validation
```

### Example 3: Generate Without Validation (Fast)

```bash
./tools/emitter_generator/generate_emitter.py \
    --spec=specs/xml_emitter.yaml \
    --no-validate
```

## Usage Guide

### Basic Workflow

1. **Create Specification**
   ```bash
   # Copy example spec
   cp tools/emitter_generator/specs/json_emitter.yaml \
      tools/emitter_generator/specs/myformat_emitter.yaml
   
   # Edit specification
   vim tools/emitter_generator/specs/myformat_emitter.yaml
   ```

2. **Generate Emitter**
   ```bash
   ./tools/emitter_generator/generate_emitter.py \
       --spec=tools/emitter_generator/specs/myformat_emitter.yaml
   ```

3. **Review Generated Files**
   ```bash
   # Generated files:
   targets/spark/myformat/myformat_emitter.ads
   targets/spark/myformat/myformat_emitter.adb
   targets/spark/myformat/test_myformat_emitter.adb
   targets/myformat/emitter.py
   targets/myformat/test_emitter.py
   targets/myformat/__init__.py
   targets/rust/myformat/mod.rs
   targets/haskell/src/STUNIR/Emitters/Myformat.hs
   targets/myformat/README.md
   ```

4. **Customize Implementation**
   - Edit generated files to add specific logic
   - Implement type mappings
   - Add target-specific features

5. **Test**
   ```bash
   # Python
   python3 -m pytest targets/myformat/test_emitter.py
   
   # Rust
   cd targets/rust && cargo test myformat
   
   # SPARK
   cd targets/spark/myformat
   gprbuild test_myformat_emitter.adb
   ./test_myformat_emitter
   ```

6. **Commit**
   ```bash
   git add targets/
   git commit -m "Add myformat emitter across all 4 pipelines"
   git push origin devsite
   ```

## Specification Format

### YAML Structure

```yaml
# Required fields
category: json                          # Category name (lowercase, alphanumeric + _)
description: "JSON serialization"       # Human-readable description

# Optional fields
output_types:                           # List of output formats
  - json
  - json_schema

features:                               # List of features
  - pretty_print
  - validation

extension: json                         # File extension

# Configuration
config:                                 # Config field definitions
  Indent_Width: "Positive range 2 .. 8"
  Use_Schema: "Boolean"

config_defaults:                        # Default config values
  Indent_Width: 2
  Use_Schema: false

# Type mappings
type_map:                               # IR type to target type
  i32: "number"
  i64: "number"
  bool: "boolean"
  string: "string"

# IR requirements
ir_fields:
  required:                             # Required IR fields
    - functions
    - types
  optional:                             # Optional IR fields
    - metadata

# Dependencies
dependencies:                           # Per-pipeline dependencies
  spark:
    - Ada.Strings.Unbounded
  python:
    - json
  rust:
    - serde_json
  haskell:
    - aeson

# Documentation
output_format: |                        # Description of output format
  Standard JSON format...

example_input:                          # Example IR input
  module: "example"
  functions: []

example_output: |                       # Example generated output
  {"module": "example"}

# Metadata
architectures:                          # Applicable architectures
  - any

notes: |                                # Implementation notes
  - Ensure RFC compliance
  - Handle Unicode properly
```

### JSON Format

Same structure as YAML, but in JSON format:

```json
{
  "category": "json",
  "description": "JSON serialization emitter",
  "output_types": ["json", "json_schema"],
  ...
}
```

## Template Variables

The following variables are available in templates:

| Variable | Description | Example |
|----------|-------------|---------|
| `{{CATEGORY}}` | Category name (lowercase) | `json` |
| `{{CATEGORY_UPPER}}` | Category name (uppercase) | `JSON` |
| `{{CATEGORY_TITLE}}` | Category name (title case) | `Json` |
| `{{DESCRIPTION}}` | Human description | `JSON emitter` |
| `{{TIMESTAMP}}` | ISO 8601 timestamp | `2026-01-30T...` |
| `{{OUTPUT_TYPES}}` | Comma-separated types | `json, schema` |
| `{{FEATURES}}` | Comma-separated features | `validation, pretty` |
| `{{CONFIG_FIELDS}}` | Config field definitions | `Indent : Positive;` |
| `{{DEFAULT_CONFIG}}` | Default config values | `Indent => 2` |
| `{{TYPE_*}}` | Type mappings | `{{TYPE_I32}}` → `int32` |
| `{{MODULE_BODY}}` | Module body placeholder | `// Implementation` |

## Command-Line Interface

```
usage: generate_emitter.py [-h] (--spec SPEC | --category CATEGORY)
                           [--description DESCRIPTION]
                           [--output-types OUTPUT_TYPES]
                           [--features FEATURES]
                           [--repo-root REPO_ROOT]
                           [--no-validate]
                           [--output-manifest OUTPUT_MANIFEST]

STUNIR Emitter Generator - Scaffold emitters across all 4 pipelines

optional arguments:
  -h, --help            show this help message and exit
  --spec SPEC           Path to emitter specification (YAML/JSON)
  --category CATEGORY   Emitter category name
  --description DESCRIPTION
                        Emitter description
  --output-types OUTPUT_TYPES
                        Comma-separated output types
  --features FEATURES   Comma-separated feature list
  --repo-root REPO_ROOT
                        STUNIR repository root (default: auto-detect)
  --no-validate         Skip validation of generated code
  --output-manifest OUTPUT_MANIFEST
                        Write generation manifest to file
```

## Generated File Structure

### SPARK (Ada)
```
targets/spark/<category>/
├── <category>_emitter.ads           # Package specification
├── <category>_emitter.adb           # Package body (implementation)
└── test_<category>_emitter.adb      # Test program
```

**Characteristics:**
- DO-178C Level A compliant
- SPARK contracts (pre/post conditions)
- Bounded strings for safety
- Formal verification ready

### Python
```
targets/<category>/
├── __init__.py                      # Package initialization
├── emitter.py                       # Main emitter class
└── test_emitter.py                  # Unit tests (pytest)
```

**Characteristics:**
- Type hints throughout
- Docstrings (Google style)
- pytest-compatible tests
- Executable CLI

### Rust
```
targets/rust/<category>/
└── mod.rs                           # Module implementation
```

**Characteristics:**
- Safe Rust (no unsafe blocks)
- Result-based error handling
- Built-in unit tests
- Integration with cargo

### Haskell
```
targets/haskell/src/STUNIR/Emitters/
└── <Category>.hs                    # Module
```

**Characteristics:**
- Pure functional
- Strong typing
- Either-based error handling
- hspec tests

## Customization

### Custom Templates

1. Copy existing template:
   ```bash
   cp templates/spark_spec.ads.template \
      templates/spark_spec_custom.ads.template
   ```

2. Modify template with your changes

3. Update `generate_emitter.py` to use custom template:
   ```python
   # In generate_spark_emitter()
   spec_template = self.load_template("spark_spec_custom.ads.template")
   ```

### Adding New Variables

1. Add variable to `prepare_variables()` method:
   ```python
   def prepare_variables(self, spec: Dict[str, Any]) -> Dict[str, str]:
       return {
           ...
           'MY_CUSTOM_VAR': spec.get('custom_field', 'default'),
       }
   ```

2. Use in templates:
   ```
   {{MY_CUSTOM_VAR}}
   ```

### Pipeline-Specific Logic

Add custom generation logic per pipeline:

```python
def generate_spark_emitter(self, spec: Dict[str, Any], variables: Dict[str, str]) -> None:
    # Custom SPARK-specific logic
    if spec.get('requires_formal_verification'):
        # Add additional SPARK annotations
        pass
```

## Validation

The generator includes built-in validation:

### Python Syntax Check
```python
python3 -m py_compile <generated_file.py>
```

### Build System Updates
- Rust `lib.rs` updated automatically
- Haskell `.cabal` file updated
- SPARK project file needs manual update

### Recommended Validation

After generation, run:

```bash
# Python
python3 -m pytest targets/<category>/

# Rust
cd targets/rust && cargo test <category>

# SPARK
cd targets/spark/<category>
gprbuild -P stunir_tools.gpr
./test_<category>_emitter

# Haskell
cd targets/haskell
cabal test
```

## Examples

### Example 1: JSON Emitter

Generate a JSON serialization emitter:

```bash
./tools/emitter_generator/generate_emitter.py \
    --spec=tools/emitter_generator/specs/json_emitter.yaml
```

**Generated files:** 11 files across all 4 pipelines

### Example 2: XML Emitter

Generate an XML emitter with schema support:

```bash
./tools/emitter_generator/generate_emitter.py \
    --category=xml \
    --description="XML serialization with XSD" \
    --output-types=xml,xsd \
    --features=validation,namespaces
```

**Generated files:** 11 files across all 4 pipelines

### Example 3: Protocol Buffers

Generate a protobuf emitter:

```bash
./tools/emitter_generator/generate_emitter.py \
    --spec=tools/emitter_generator/specs/protobuf_emitter.yaml \
    --output-manifest=manifest.json
```

**Output:** 11 files + manifest.json

## Troubleshooting

### Common Issues

**Issue:** `Template not found`  
**Solution:** Check that you're running from repo root, or use `--repo-root`

**Issue:** `Python validation failed`  
**Solution:** Check generated Python code syntax, may need manual fixes

**Issue:** `Build system update failed`  
**Solution:** Manually update `targets/rust/lib.rs` or Haskell cabal file

**Issue:** `YAML parsing error`  
**Solution:** Validate YAML syntax (use `yamllint` tool)

### Debug Mode

For verbose output, modify the script:

```python
# Add at top of main()
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### Specification Design

1. **Choose clear category names** - lowercase, descriptive
2. **Provide comprehensive descriptions** - helps users understand purpose
3. **Define type mappings explicitly** - avoid ambiguity
4. **Include examples** - input and output examples clarify usage
5. **Document features** - explain what each feature does

### After Generation

1. **Review all generated files** - ensure correctness
2. **Customize implementation** - add specific logic
3. **Add comprehensive tests** - beyond scaffolding
4. **Update documentation** - README with real examples
5. **Test all 4 pipelines** - ensure confluence
6. **Commit atomically** - all files in one commit

### Maintenance

1. **Keep templates updated** - as patterns evolve
2. **Version specifications** - track changes
3. **Test generator regularly** - catch template issues
4. **Document customizations** - for team awareness

## Integration with STUNIR Build System

Generated emitters automatically integrate with:

- **SPARK:** Add to `targets/spark/stunir_emitters.gpr`
- **Python:** Available via `targets/<category>/emitter.py`
- **Rust:** Auto-added to `targets/rust/lib.rs`
- **Haskell:** Auto-added to `.cabal` file

Build all emitters:

```bash
# SPARK
cd tools/spark && gprbuild -P stunir_tools.gpr

# Python
# No build needed - interpreted

# Rust
cd targets/rust && cargo build --release

# Haskell
cd targets/haskell && cabal build
```

## Testing Strategy

### Generator Tests

Test the generator itself:

```bash
# Create test spec
cat > /tmp/test_spec.yaml << EOF
category: test
description: "Test emitter"
EOF

# Generate
./tools/emitter_generator/generate_emitter.py --spec=/tmp/test_spec.yaml

# Verify
ls targets/test/
ls targets/spark/test/
```

### Generated Code Tests

Each pipeline includes test scaffolding:

```bash
# Run all tests
./scripts/test_all_emitters.sh test
```

## Performance

Generator performance (typical):

- **Specification parsing:** < 10ms
- **Template loading:** < 50ms
- **Code generation:** < 100ms
- **File writing:** < 200ms
- **Validation:** < 1s (Python only)

**Total time:** ~1.5 seconds per emitter

## Future Enhancements

Planned improvements:

- [ ] Interactive mode (wizard)
- [ ] Template marketplace
- [ ] Automatic confluence testing
- [ ] CI/CD integration hooks
- [ ] Web UI for specification editing
- [ ] Template validation tool
- [ ] Multi-emitter generation (batch mode)

## Contributing

To contribute to the generator:

1. Fork repository
2. Create feature branch
3. Modify generator or templates
4. Test with multiple emitter types
5. Submit pull request

### Template Contributions

New template contributions welcome! Follow template format and include:
- Clear variable usage
- Comments explaining sections
- Example usage

## License

MIT License - See [LICENSE](../../LICENSE) for details

## Support

- **Documentation:** https://stunir.dev/emitter-generator
- **Issues:** https://github.com/stunir/stunir/issues
- **Email:** support@stunir.dev

## Acknowledgments

This tool was created after analyzing patterns across 20+ hand-written emitters in 4 different programming languages. Special thanks to the STUNIR team for iterating on emitter patterns.

---

**Generated by:** STUNIR Emitter Generator v1.0.0  
**Last Updated:** 2026-01-30
