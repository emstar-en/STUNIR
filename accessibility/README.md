# STUNIR Accessibility

Accessibility standards and guidelines for STUNIR toolchain outputs.

## Overview

This directory contains accessibility standards, guidelines, and validation tools
to ensure STUNIR-generated outputs meet accessibility requirements.

## Contents

- `standards/` - Accessibility standards documentation and checkers
  - `WCAG_compliance.md` - WCAG 2.1 compliance guidelines
  - `standards_checker.py` - Automated accessibility checker
  - `README.md` - Standards documentation

## Quick Start

```bash
# Run accessibility check on generated output
python3 accessibility/standards/standards_checker.py --file output.html

# Check multiple files
python3 accessibility/standards/standards_checker.py --dir ./outputs/

# Generate accessibility report
python3 accessibility/standards/standards_checker.py --report
```

## Standards Covered

1. **WCAG 2.1** - Web Content Accessibility Guidelines
2. **ARIA** - Accessible Rich Internet Applications
3. **Documentation** - Code documentation accessibility
4. **Output Formats** - Accessible output format requirements

## Integration

The accessibility module integrates with the STUNIR validation pipeline:

```
pipeline
  → target_emit
  → accessibility_check  # This module
  → manifest_gen
  → verify
```

## Issue Reference

This module resolves: `accessibility/standards/1160`
