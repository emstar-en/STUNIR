# Directory Cleanup and Organization Recommendations

## Current Issues

1. **Root directory clutter**: 125+ items making navigation difficult
2. **Scattered Python scripts**: Analysis/utility scripts mixed with core code
3. **Multiple doc formats**: .md, .pdf, .json reports scattered
4. **Unclear boundaries**: Hard to distinguish between:
   - Core implementation (Ada SPARK)
   - Utilities (Python)
   - Documentation
   - Build artifacts
   - Examples/tests

## Recommended Restructuring

```
stunir/
├── src/                    # All source code
│   ├── ada/               # Ada SPARK implementation
│   │   ├── core/          # Core packages (moved from root/core)
│   │   └── targets/       # Target emitters (moved from root/targets)
│   └── python/            # Python utilities (consolidated)
│
├── docs/                   # All documentation
│   ├── guides/            # User guides, tutorials
│   ├── api/               # API documentation
│   ├── specs/             # Specifications
│   └── reports/           # Analysis/completion reports
│
├── tests/                  # All test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── golden/            # Golden master tests
│
├── tools/                  # Development tools
│   ├── spark/             # Ada SPARK build tools
│   ├── scripts/           # Utility scripts (python)
│   └── analysis/          # Analysis tools
│
├── examples/               # Example code and demos
│
├── build/                  # Build outputs (gitignored)
│
├── local_tasks/            # Local model workspace
│
└── [root files]           # Only essential: README, LICENSE, etc.
```

## Priority Cleanup Actions

### High Priority (Do Now)

1. **Move Python utilities**
   ```
   *.py → tools/scripts/
   analysis/*.py → tools/analysis/
   ```

2. **Consolidate docs**
   ```
   *.md (except README) → docs/guides/
   *.pdf → docs/guides/
   *_report.json → docs/reports/
   ```

3. **Move core Ada code**
   ```
   core/ → src/ada/core/
   targets/ → src/ada/targets/
   ```

### Medium Priority

4. **Organize examples**
   ```
   examples/ → examples/ (but clean up)
   inputs/ → examples/inputs/
   ```

5. **Create .gitignore sections**
   - Ignore build/, dist/, *.o, *.ali
   - Ignore local_tasks/work/ (working files)

### Low Priority

6. **Archive old artifacts**
   - Move old analysis reports to docs/reports/archive/
   - Remove temporary/generated files

## Benefits for Local Models

After cleanup:
- **Faster file search**: Less directory noise
- **Clearer context**: Obvious where to look for types/interfaces
- **Predictable paths**: `src/ada/targets/spark/embedded/`
- **Focused working directory**: `local_tasks/` stays clean

## Suggested .gitignore Additions

```gitignore
# Build artifacts
build/
dist/
*.o
*.ali
*.exe
b~*.ad[sb]

# Local model workspace
local_tasks/work/*
!local_tasks/work/.gitkeep
local_tasks/tests/temp/

# Python
__pycache__/
*.pyc
.pytest_cache/

# Temp files
*.tmp
*.log
*~
```

## Impact on Existing Scripts

After restructuring, update:
- Build scripts to use `src/ada/` paths
- Import paths in Python scripts
- Documentation references
- CI/CD pipeline paths

## Rollout Strategy

1. **Create new structure** (don't move yet)
2. **Test with local model** on focused task
3. **Gradually migrate** one category at a time
4. **Update references** after each migration
5. **Archive old structure** when complete

## For Immediate Use

You can start using `local_tasks/` right away without restructuring. The cleanup is recommended but not required for local model delegation to work.
