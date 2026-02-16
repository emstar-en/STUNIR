# Local Model Task System - Quick Start

## You Now Have:

✅ **local_tasks/** - Dedicated workspace for local model delegation
✅ **LOCAL_MODEL_GUIDE.md** - Complete strategy for task delegation
✅ **CLEANUP_RECOMMENDATIONS.md** - Directory organization suggestions
✅ **First Task Ready** - Arithmetic statement emission (in context/current_task.md)

## Immediate Next Steps

### Option 1: Try the First Task with Your Local Model

**Hand this to Mistral 3 14B:**

```
You are working on an Ada SPARK embedded C code emitter. Read the task in 
local_tasks/context/current_task.md and implement the arithmetic operations.

Context files:
- local_tasks/context/emitter_types_reference.ads (type definitions)
- local_tasks/context/example_arithmetic_output.txt (expected output)

Target file: targets/spark/embedded/embedded_emitter.adb (lines ~200-250)

Task: Add Stmt_Add, Stmt_Sub, Stmt_Mul, Stmt_Div cases to Emit_Statement procedure.
```

Estimated completion: 30-45 minutes for local model

### Option 2: Continue with Cloud Model

**Next priorities for cloud models:**

1. **Complete remaining emitter targets** (25+ incomplete)
   - Assembly (x86, ARM, MIPS, RISC-V)
   - Mobile (Swift, Kotlin)
   - Scientific (Fortran, Julia, R)

2. **Add comprehensive test suites**
   - Unit tests for each emitter
   - Integration tests
   - Golden master tests

3. **Implement formal verification**
   - Add SPARK contracts to all procedures
   - Prove absence of runtime errors
   - DO-178C compliance checks

4. **Directory restructuring** (see CLEANUP_RECOMMENDATIONS.md)

## Mistral 3 14B Configuration

**Recommended Settings:**
```
model: mistral-3-14b-instruct-q4
context_length: 8192  # Safe for Q4
temperature: 0.1      # Low for code generation
top_p: 0.95
max_tokens: 2048      # Reserve for output
```

**Context Budget Per Task:**
- Task description: ~500 tokens
- Type definitions: ~800 tokens
- Current code section: ~1500 tokens
- Examples: ~500 tokens
- Output buffer: ~2000 tokens
- **Total: ~5300 tokens** (comfortable margin)

## Task Completion Workflow

### For You (orchestrating):

1. **Prepare Task** (Cloud model)
   - Identify self-contained unit of work
   - Extract minimal context (types, examples)
   - Create task specification in local_tasks/context/
   - Estimate: 10-15 minutes per task

2. **Execute Task** (Local model)
   - Local model reads context files
   - Implements changes
   - Runs compilation test
   - Estimate: 30-60 minutes per task

3. **Integrate & Verify** (Cloud model)
   - Review local model output
   - Run full test suite
   - Handle any issues
   - Commit changes
   - Estimate: 5-10 minutes per task

### Cost-Benefit Analysis

**Local Model Tasks (Mistral 3 14B):**
- Cost: ~$0 (local inference)
- Speed: 30-60 min/task
- Quality: Good for focused, pattern-based work
- Best for: Repetitive implementation, single-function fixes

**Cloud Model Tasks (GPT-4/Claude):**
- Cost: $0.10-0.50/task
- Speed: 2-10 min/task
- Quality: Excellent for complex reasoning
- Best for: Architecture, debugging, integration

**Hybrid Approach Savings:**
- 20 focused tasks → Local model: $0 vs Cloud: ~$5-10
- 5 complex tasks → Cloud model: ~$2
- **Total savings: ~70-80% on focused implementation work**

## Next Available Tasks (Queued for Local Models)

All tasks are sized for 6K token context, 30-60 min completion:

### Priority 1: Complete Embedded Emitter
1. ✅ **Arithmetic operations** (current_task.md) - READY NOW
2. Add comparison operations (Eq, Ne, Lt, Gt, Le, Ge) - ~100 lines
3. Add control flow (If, Loop) - ~150 lines
4. Add function calls - ~100 lines

### Priority 2: Assembly Emitters
5. x86_64 basic instructions (mov, add, sub, ret) - ~150 lines
6. ARM basic instructions - ~150 lines
7. RISC-V basic instructions - ~150 lines

### Priority 3: Test Infrastructure
8. Embedded emitter unit tests - ~200 lines
9. Type conversion tests - ~100 lines
10. Error handling tests - ~100 lines

### Priority 4: Boilerplate Generation
11. Equality functions for remaining types - ~50 lines each
12. SPARK contracts for procedures - ~30 lines each
13. Documentation comments - varies

## Directory Cleanup Status

**Current:** 125+ items in root (cluttered)
**Recommended:** See CLEANUP_RECOMMENDATIONS.md

**Decision needed:** Clean up now or later?
- **Now**: Better for local model navigation
- **Later**: Can work with current structure

## Metrics to Track

For your local model delegation:

| Metric | Target | Why |
|--------|--------|-----|
| Task size | < 200 lines code | Fits in context |
| Context size | < 6K tokens | Safe margin for Q4 |
| Completion rate | > 80% | Validates task sizing |
| Integration issues | < 20% | Validates preparation |
| Compile on first try | > 60% | Validates examples |

## Success Criteria

**Local model task is well-designed if:**
1. ✅ Local model completes without asking questions
2. ✅ Code compiles on first or second try
3. ✅ Follows existing patterns correctly
4. ✅ Completes in single session (no continuation needed)
5. ✅ Integration requires < 10 min of fixes

## Getting Started Right Now

**Try this workflow:**

1. Start your local Mistral 3 14B model
2. Give it the prompt above with current_task.md
3. Let it work for 30-45 minutes
4. Review output with me (cloud model)
5. I'll integrate and prepare next task

**Or continue with me for:**
- Complex debugging
- Architecture decisions
- Multi-file refactoring
- Planning next 10 tasks

## Questions?

Ask me to:
- Generate more local model tasks
- Help integrate local model output
- Debug compilation issues
- Plan larger features
- Restructure directories
- Anything requiring broad codebase understanding

---

**Bottom Line:** 
- Local models: Focused, isolated, pattern-based implementation
- Cloud models: Planning, integration, complex reasoning, debugging
- Hybrid: 70-80% cost savings with same quality
