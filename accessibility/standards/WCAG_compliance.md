# WCAG 2.1 Compliance Guidelines for STUNIR

## Overview

This document outlines WCAG 2.1 compliance requirements for STUNIR-generated outputs.

## Compliance Levels

| Level | Description | STUNIR Target |
|-------|-------------|---------------|
| A | Minimum accessibility | Required |
| AA | Enhanced accessibility | Recommended |
| AAA | Highest accessibility | Optional |

## STUNIR-Specific Guidelines

### 1. Perceivable

#### 1.1 Text Alternatives
- All non-text content must have text alternatives
- Generated code comments should describe functionality
- Diagrams/charts must include alt text or descriptions

#### 1.2 Time-based Media
- Not typically applicable to STUNIR outputs

#### 1.3 Adaptable
- Generated HTML must use semantic markup
- Data tables must have proper headers
- Reading order must be logical

#### 1.4 Distinguishable
- Color must not be the only means of conveying information
- Text contrast ratio: 4.5:1 minimum (Level AA)
- Text resizable to 200% without loss of content

### 2. Operable

#### 2.1 Keyboard Accessible
- All functionality available via keyboard
- No keyboard traps

#### 2.4 Navigable
- Page titles descriptive and unique
- Heading hierarchy logical (h1 → h2 → h3)
- Link purpose clear from context

### 3. Understandable

#### 3.1 Readable
- Language of page identified
- Unusual words/abbreviations defined

#### 3.2 Predictable
- Navigation consistent
- Components behave predictably

#### 3.3 Input Assistance
- Error identification clear
- Labels descriptive

### 4. Robust

#### 4.1 Compatible
- Valid HTML/CSS
- ARIA used correctly when needed
- Compatible with assistive technologies

## STUNIR Output Requirements

### Generated HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Descriptive Title - STUNIR Output</title>
</head>
<body>
  <!-- Proper heading hierarchy -->
  <h1>Main Heading</h1>
  <main>
    <!-- Semantic structure -->
    <article>
      <h2>Section Heading</h2>
      <p>Content...</p>
    </article>
  </main>
</body>
</html>
```

### Generated Documentation

- Use clear, simple language
- Provide code examples with comments
- Include table of contents for long documents
- Use consistent terminology

### Generated Code

- Include descriptive comments
- Use meaningful variable/function names
- Provide accessibility metadata where applicable

## Validation

Use the `standards_checker.py` tool to validate outputs:

```bash
python3 standards_checker.py --wcag-level AA output.html
```

## References

- [WCAG 2.1 Specification](https://www.w3.org/TR/WCAG21/)
- [WCAG Quick Reference](https://www.w3.org/WAI/WCAG21/quickref/)
- [WebAIM](https://webaim.org/)
