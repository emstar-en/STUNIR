# Local Model Task Guide

## Context Window: Mistral 3 14B Q4
- **Recommended Max Context**: 8,192 tokens (~2,000 lines)
- **Safe Working Context**: 6,000 tokens (~1,500 lines)
- **Output Reserve**: 2,000 tokens (~500 lines)

## Task Structure

Each task for local models should follow this structure:

```
local_tasks/
├── context/
│   ├── emitter_types.ads      # Core type definitions (REFERENCE ONLY)
│   ├── current_task.md        # Clear, focused task description
│   └── example_output.adb     # Example of expected output
├── work/
│   └── [single_file.adb]      # File to edit (max 200 lines)
├── tests/
│   └── test_single_file.adb   # Minimal test
└── build/
    └── compile.sh             # Simple compilation command
```

## Task Types Suitable for Local Models

### ✅ GOOD for Local Models

1. **Single Function Implementation**
   - Context: Type definitions + function signature
   - Task: Implement 1-2 functions in isolation
   - Lines: 50-150 lines of new code

2. **Pattern-Based Code Generation**
   - Context: 2-3 examples + pattern description
   - Task: Generate similar code following pattern
   - Lines: 100-200 lines

3. **Isolated Bug Fix**
   - Context: Failing function + error message + expected behavior
   - Task: Fix specific bug
   - Lines: 10-50 line changes

4. **Test Case Writing**
   - Context: Function signature + behavior description
   - Task: Write test cases
   - Lines: 50-100 lines

5. **Documentation**
   - Context: Code snippet
   - Task: Add comments/contracts
   - Lines: Same as code + 20-30% comments

### ❌ AVOID for Local Models

1. **Multi-file refactoring** (needs full project context)
2. **Architecture decisions** (needs broad understanding)
3. **Complex debugging** (requires exploring many files)
4. **Integration tasks** (needs understanding of interfaces)
5. **Stack overflow issues** (requires deep system knowledge)

## Example Task Workflow

### Cloud Model (Initial Planning)
```bash
# 1. Analyze the full codebase
# 2. Identify independent, self-contained task
# 3. Extract minimal required context
# 4. Create task specification
# 5. Prepare examples and expected output
```

### Local Model (Execution)
```bash
# 1. Read context/current_task.md
# 2. Review context/emitter_types.ads for type info
# 3. Look at context/example_output.adb for pattern
# 4. Edit work/[file]
# 5. Run build/compile.sh to verify
# 6. Report completion or issues
```

### Cloud Model (Integration)
```bash
# 1. Review local model output
# 2. Run full test suite
# 3. Handle any integration issues
# 4. Commit changes
```

## Next Available Tasks (Prioritized for Local Models)

### Priority 1: Complete Embedded Emitter Statement Handlers

**Task**: Implement statement emission for arithmetic operations
- **File**: targets/spark/embedded/embedded_emitter.adb
- **Lines**: ~100 lines (isolated case statements)
- **Context Needed**: 
  - Embedded_Statement type definition
  - Get_C_Type function
  - Example of Stmt_Var_Decl handling
- **Estimated Time**: 30-60 minutes

### Priority 2: Assembly Emitter (x86_64)

**Task**: Implement basic x86_64 instruction emission
- **File**: targets/spark/assembly/x86_64_emitter.adb
- **Lines**: ~150 lines (focused on mov, add, sub, ret)
- **Context Needed**:
  - Register definitions
  - Instruction format
  - Assembly syntax reference

### Priority 3: Add Test Cases

**Task**: Create test cases for embedded emitter
- **File**: targets/spark/embedded/embedded_emitter_tests.adb
- **Lines**: ~100 lines
- **Context Needed**:
  - Test framework structure
  - Embedded_Emitter interface
  - Expected output examples

### Priority 4: Implement Missing Equality Functions

**Task**: Add equality functions for remaining vector types
- **File**: targets/spark/embedded/embedded_emitter.adb
- **Lines**: ~50 lines (multiple small functions)
- **Context Needed**:
  - Example equality function
  - Record definitions

## File Size Guidelines

For local models, keep files small and focused:

- **Specification (.ads)**: Max 200 lines
- **Body (.adb)**: Max 300 lines per file
- **If larger**: Split into multiple packages

## Context Optimization Tips

1. **Extract Core Types**: Create minimal type definition file
2. **Remove Comments**: Strip comments from reference code to save tokens
3. **Focus Examples**: Show only 2-3 relevant examples, not entire file
4. **Inline Dependencies**: Copy short dependency definitions into task context
5. **Use Pseudocode**: Describe complex algorithms in pseudocode first

## Success Metrics

A well-structured local model task should:
- ✅ Fit in 6,000 tokens total context
- ✅ Have clear pass/fail criteria
- ✅ Compile independently or with 1-2 dependencies
- ✅ Complete in single session (no need to "continue")
- ✅ Produce output verifiable by simple test
