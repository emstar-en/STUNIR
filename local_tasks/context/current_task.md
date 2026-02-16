# Task: Implement Arithmetic Statement Emission

## Objective
Implement the emission of arithmetic operations (Add, Sub, Mul, Div) in the embedded emitter's `Emit_Statement` procedure.

## Context
You are working on `embedded_emitter.adb` line ~150-250 in the `Emit_Statement` procedure. Currently, only `Stmt_Var_Decl` and `Stmt_Return` are implemented. You need to add cases for arithmetic operations.

## File Location
- **Work File**: `targets/spark/embedded/embedded_emitter.adb`
- **Specification**: `targets/spark/embedded/embedded_emitter.ads`
- **Reference**: `context/emitter_types.ads` (for type definitions)

## Task Details

### Input
An `Embedded_Statement` record with:
- `Stmt_Type`: One of `Stmt_Add`, `Stmt_Sub`, `Stmt_Mul`, `Stmt_Div`
- `Target`: Variable name receiving result (Identifier_String)
- `Left_Op`: Left operand variable name (Identifier_String)
- `Right_Op`: Right operand variable name (Identifier_String)
- `Data_Type`: Result type (IR_Data_Type)

### Expected Output (C code)
For a statement like:
```ada
Stmt := (
   Stmt_Type => Stmt_Add,
   Data_Type => Type_I32,
   Target    => "result",
   Left_Op   => "a",
   Right_Op  => "b",
   Value     => ""
)
```

Should emit C code:
```c
result = a + b;
```

### Implementation Requirements

1. Add four new `when` cases in the `Emit_Statement` case statement:
   - `when Stmt_Add =>`
   - `when Stmt_Sub =>`
   - `when Stmt_Mul =>`
   - `when Stmt_Div =>`

2. Each case should:
   - Append indentation spaces (`Indent * 3` spaces)
   - Append target variable name
   - Append ` = `
   - Append left operand
   - Append operator (` + `, ` - `, ` * `, ` / `)
   - Append right operand
   - Append `;\n`
   - Check for buffer overflow after each append

3. Use the helper function `Append_To_Content` which:
   - Takes `Content : in out Content_String`
   - Takes `Text : in String`
   - Returns `Status : out Emitter_Status`
   - Sets Status to `Error_Buffer_Overflow` if content is full

## Example Pattern (from existing Stmt_Return)

```ada
when Stmt_Return =>
   Append_To_Content (Content, Indent_Str, Status);
   if Status /= Success then
      return;
   end if;
   
   Append_To_Content (Content, "return ", Status);
   if Status /= Success then
      return;
   end if;
   
   if Identifier_Strings.Length (Stmt.Value) > 0 then
      Append_To_Content (Content, 
         Identifier_Strings.To_String (Stmt.Value), Status);
      if Status /= Success then
         return;
      end if;
   end if;
   
   Append_To_Content (Content, ";" & New_Line, Status);
```

## Success Criteria

1. ✅ Code compiles without errors
2. ✅ All four arithmetic operations implemented
3. ✅ Proper error checking after each Append_To_Content
4. ✅ Correct C syntax generated
5. ✅ Follows existing code style

## Testing

Run this after implementation:
```bash
cd targets/spark/embedded
wsl gnatmake -I.. embedded_emitter.adb
```

Expected: No compilation errors

## Hints

- Look at lines 150-250 in `embedded_emitter.adb` for the case statement
- The `New_Line` constant is defined at the top of the file
- Indentation is typically 3 spaces per level
- Don't forget to convert Identifier_String to String using `Identifier_Strings.To_String()`
- Early return on any error status

## Estimated Effort
30-45 minutes

## Dependencies
- See `context/emitter_types.ads` for type definitions
- See `context/example_stmt_add.txt` for expected output format
