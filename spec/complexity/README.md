# STUNIR Spec Complexity Analyzer

**Pipeline Stage:** spec → complexity  
**Issue:** #1150

## Overview

Analyzes complexity metrics of STUNIR specifications to help maintain
code quality and identify areas needing refactoring.

## Files

| File | Description |
|------|-------------|
| `analyzer.py` | Complexity analysis engine |
| `metrics.json` | Metrics configuration and thresholds |

## Usage

### CLI

```bash
# Analyze a spec
python analyzer.py spec.json

# Analyze multiple specs
python analyzer.py spec1.json spec2.json --json

# Compare two specs
python analyzer.py --compare old_spec.json new_spec.json

# Save results
python analyzer.py spec.json --json -o results.json
```

### Python API

```python
from analyzer import ComplexityAnalyzer, ComplexityMetrics
from pathlib import Path

analyzer = ComplexityAnalyzer()

# Analyze spec dictionary
metrics = analyzer.analyze(spec_dict)
print(f"Complexity: {metrics.overall_complexity}")
print(f"Maintainability: {metrics.maintainability_index}")

# Analyze file
result = analyzer.analyze_file(Path('spec.json'))

# Compare metrics
diff = analyzer.compare(old_metrics, new_metrics)
```

## Metrics

### Basic Counts
- `function_count` - Number of functions
- `type_count` - Number of type definitions
- `import_count` - Number of imports
- `export_count` - Number of exports
- `constant_count` - Number of constants

### Complexity Scores
- `cyclomatic_complexity` - Decision point complexity
- `nesting_depth` - Maximum nesting level
- `type_complexity` - Aggregate type complexity

### Derived Scores
- `overall_complexity` - Weighted sum of metrics
- `maintainability_index` - 0-100 score (higher is better)

## Complexity Categories

| Category | Max Score | Description |
|----------|-----------|-------------|
| Trivial | 5 | Minimal logic |
| Simple | 15 | Basic structures |
| Moderate | 30 | Multiple types/functions |
| Complex | 60 | Requires careful review |
| Critical | 60+ | Consider refactoring |
