# STUNIR Accessibility Standards

This directory contains accessibility standards and validation tools for STUNIR.

## Files

| File | Description |
|------|-------------|
| `WCAG_compliance.md` | WCAG 2.1 compliance guidelines |
| `standards_checker.py` | Automated accessibility validation tool |
| `aria_guidelines.md` | ARIA implementation guidelines |

## Usage

### Command Line

```bash
# Validate a single file
python3 standards_checker.py file.html

# Validate directory
python3 standards_checker.py --dir ./output/

# JSON output
python3 standards_checker.py --json file.html
```

### Python API

```python
from standards_checker import AccessibilityChecker

checker = AccessibilityChecker()
result = checker.check_file('output.html')
print(f"Accessible: {result.valid}")
```

## Standards Reference

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Practices](https://www.w3.org/WAI/ARIA/apg/)

## Issue Reference

Resolves: `accessibility/standards/1160`
