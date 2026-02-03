# AI Memory Bins (The "Swap Space")

## 🤖 Instructions for AI Agents

This directory (`meta/ai_bins/`) is your **Persistent Memory**.
Because you are stateless between turns, you must use these files to store your context, plans, and drafts.

### The Bins

- **`A_TASK.md` (Focus)**: ALWAYS read this first. It contains your current objective and progress checklist. Update it as you complete steps.
- **`B_CONTEXT.md` (Working Set)**: List the files you are currently modifying here. This helps you (and future you) know where to look.
- **`C_DRAFT.md` (Workbench)**: Write your code snippets here FIRST. Verify them, then move them to the actual source files.
- **`D_ERRORS.md` (Debugger)**: Paste build errors or test failures here. Analyze them in this file before proposing a fix.
- **`E_DECISIONS.md` (Ledger)**: When you make a significant architectural choice (e.g., "We are using X instead of Y"), record it here.

### Rules
1.  **Read-Modify-Write**: In every session, check `A_TASK.md` to orient yourself.
2.  **Keep it Clean**: Don't dump huge files here. Use references (paths) in `B_CONTEXT.md`.
3.  **Update State**: Before you finish your turn, update `A_TASK.md` to reflect what you just did.
