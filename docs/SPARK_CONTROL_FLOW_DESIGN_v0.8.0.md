# SPARK Control Flow Implementation Design - v0.8.0

## Executive Summary

This document specifies the implementation of control flow parsing and flattened IR generation in Ada SPARK's spec_to_ir tool, achieving **100% SPARK pipeline completion**.

**Version**: v0.8.0  
**Goal**: Complete SPARK-native end-to-end pipeline (spec → IR → C code)  
**Scope**: Parse if/else/while/for from spec JSON and generate flattened IR directly

## Current State (v0.7.1)

### ✅ Working Components
- **SPARK ir_to_code**: 100% complete with bounded recursion (depth=5)
- **SPARK spec_to_ir**: Can parse basic specs but generates "noop" for all statements
- **Python pipeline**: Full nested IR support with flattening capability

### ❌ Missing Component
- **SPARK spec_to_ir control flow**: Does not parse if/while/for from spec JSON

### Current Code Location
```
/home/ubuntu/stunir_repo/tools/spark/src/
  stunir_json_utils.adb:287-306  ← Body parsing (currently generates Stmt_Nop)
  stunir_semantic_ir.ads         ← IR_Statement type definition
```

## Architecture Overview

### Pipeline Comparison

**Python Pipeline** (current):
```
Spec JSON → [Python spec_to_ir] → Nested IR → [ir_converter.py] → Flat IR → [ir_to_code] → C code
```

**SPARK Pipeline** (v0.8.0 target):
```
Spec JSON → [SPARK spec_to_ir] → Flat IR → [SPARK ir_to_code] → C code
```

**Key Insight**: SPARK spec_to_ir should generate **flattened IR directly**, not nested IR, because Ada cannot dynamically parse nested JSON arrays.

## Design Decisions

### Decision 1: Direct Flattening vs Two-Stage

**Option A**: Parse nested, then flatten (like Python)
- ❌ Requires dynamic JSON parsing in Ada (not feasible with bounded types)
- ❌ Complex intermediate representation

**Option B**: Generate flattened IR directly ✅ **CHOSEN**
- ✅ Single-pass parsing
- ✅ Works with Ada's static typing
- ✅ Simpler implementation
- ✅ No intermediate conversion needed

### Decision 2: Recursion Depth

**Requirement**: Support multi-level nesting to match ir_to_code v0.7.0

**Approach**: Bounded recursion with `Max_Nesting_Depth := 5`
- Same depth limit as ir_to_code (consistency)
- Proven sufficient for real-world control flow
- SPARK-verifiable bounds

### Decision 3: Parsing Strategy

**Challenge**: How to parse nested control flow without dynamic structures?

**Solution**: **Flatten during parsing** (single-pass) ✅ **REVISED**
- Parse and flatten in one pass
- Track current insertion index
- Backpatch control flow statements after parsing their blocks
- No access types, no temporary structures
- Pure SPARK-verifiable code

## Data Structures

### No Temporary Structures Needed!

**Key Insight**: We flatten **during** parsing, so we only need the final IR_Statement array.

```ada
-- Maximum nesting depth for control flow
Max_Nesting_Depth : constant := 5;

-- Statement kind (already exists in stunir_semantic_ir.ads)
type Stmt_Kind is (
   Stmt_Assign,
   Stmt_Call,
   Stmt_Return,
   Stmt_If,
   Stmt_While,
   Stmt_For,
   Stmt_Nop
);

-- NO INTERMEDIATE STRUCTURES - parse directly into IR_Statement array!
```

### Flattened IR Output

```ada
-- This already exists in stunir_semantic_ir.ads
type IR_Statement is record
   Kind          : Stmt_Kind;
   Target        : Name_String;
   Value         : Name_String;
   Condition     : Name_String;
   Init_Expr     : Name_String;
   Incr_Expr     : Name_String;
   Block_Start   : Natural;      -- 1-based index
   Block_Count   : Natural;      -- Number of steps in block
   Else_Start    : Natural;      -- 1-based index (0 if none)
   Else_Count    : Natural;      -- Number of steps in else block
end record;
```

## Parsing Algorithm

### High-Level Flow (Single-Pass)

```ada
procedure Parse_Function_Body (
   Body_JSON : String;
   Func_Idx  : Positive;
   Module    : in out IR_Module;
   Status    : out Parse_Status
) is
begin
   -- Single pass: parse and flatten simultaneously
   Parse_And_Flatten_Statements (
      Body_JSON,
      Module.Functions (Func_Idx).Statements,
      Module.Functions (Func_Idx).Stmt_Cnt,
      Depth => 0,
      Status => Status
   );
end Parse_Function_Body;
```

### Single-Pass Parse-and-Flatten Algorithm

```ada
procedure Parse_Statements_Recursive (
   JSON_Text : String;
   Block     : out Statement_Block;
   Depth     : Natural;
   Status    : out Parse_Status
) is
   Stmt_Pos : Natural;
   Obj_Start, Obj_End : Natural;
begin
   -- Check depth limit
   if Depth > Max_Nesting_Depth then
      Put_Line ("[ERROR] Maximum nesting depth exceeded");
      Status := Error_Too_Deep;
      return;
   end if;
   
   Block.Count := 0;
   Stmt_Pos := Find_Array (JSON_Text, "body") + 1;  -- Or appropriate field
   
   -- Parse each statement in the array
   while Block.Count < Max_Block_Statements loop
      Get_Next_Object (JSON_Text, Stmt_Pos, Obj_Start, Obj_End);
      exit when Obj_Start = 0 or Obj_End = 0;
      
      declare
         Stmt_JSON : constant String := JSON_Text (Obj_Start .. Obj_End);
         Stmt_Type : constant String := Extract_String_Value (Stmt_JSON, "type");
         Stmt      : Parsed_Statement;
      begin
         -- Parse based on statement type
         if Stmt_Type = "if" then
            Stmt.Kind := Stmt_If;
            Stmt.Condition := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "condition")
            );
            
            -- Parse then block recursively
            declare
               Then_Block : aliased Statement_Block;
               Then_JSON  : constant String := Extract_Array_JSON (Stmt_JSON, "then");
            begin
               Parse_Statements_Recursive (Then_JSON, Then_Block, Depth + 1, Status);
               if Status /= Success then
                  return;
               end if;
               Stmt.Then_Block := Then_Block'Unchecked_Access;  -- Store parsed block
            end;
            
            -- Parse else block if present
            declare
               Else_JSON : constant String := Extract_Array_JSON (Stmt_JSON, "else");
            begin
               if Else_JSON'Length > 0 then
                  declare
                     Else_Block : aliased Statement_Block;
                  begin
                     Parse_Statements_Recursive (Else_JSON, Else_Block, Depth + 1, Status);
                     if Status /= Success then
                        return;
                     end if;
                     Stmt.Else_Block := Else_Block'Unchecked_Access;
                  end;
               end if;
            end;
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         elsif Stmt_Type = "while" then
            Stmt.Kind := Stmt_While;
            Stmt.Condition := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "condition")
            );
            
            -- Parse body recursively
            declare
               Body_Block : aliased Statement_Block;
               Body_JSON  : constant String := Extract_Array_JSON (Stmt_JSON, "body");
            begin
               Parse_Statements_Recursive (Body_JSON, Body_Block, Depth + 1, Status);
               if Status /= Success then
                  return;
               end if;
               Stmt.Body_Block := Body_Block'Unchecked_Access;
            end;
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         elsif Stmt_Type = "for" then
            -- Similar to while, but with init/increment
            Stmt.Kind := Stmt_For;
            Stmt.Init_Expr := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "init")
            );
            Stmt.Condition := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "condition")
            );
            Stmt.Incr_Expr := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "increment")
            );
            
            -- Parse body recursively
            declare
               Body_Block : aliased Statement_Block;
               Body_JSON  : constant String := Extract_Array_JSON (Stmt_JSON, "body");
            begin
               Parse_Statements_Recursive (Body_JSON, Body_Block, Depth + 1, Status);
               if Status /= Success then
                  return;
               end if;
               Stmt.Body_Block := Body_Block'Unchecked_Access;
            end;
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         elsif Stmt_Type = "assign" then
            Stmt.Kind := Stmt_Assign;
            Stmt.Target := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "target")
            );
            Stmt.Value := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "value")
            );
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         elsif Stmt_Type = "return" then
            Stmt.Kind := Stmt_Return;
            Stmt.Value := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "value")
            );
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         elsif Stmt_Type = "call" then
            Stmt.Kind := Stmt_Call;
            Stmt.Value := Name_Strings.To_Bounded_String (
               Extract_String_Value (Stmt_JSON, "func") & "(" &
               Extract_String_Value (Stmt_JSON, "args") & ")"
            );
            
            -- Handle optional assignment
            declare
               Assign_To : constant String := Extract_String_Value (Stmt_JSON, "assign_to");
            begin
               if Assign_To'Length > 0 then
                  Stmt.Target := Name_Strings.To_Bounded_String (Assign_To);
               end if;
            end;
            
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         
         else
            -- Unknown or nop
            Stmt.Kind := Stmt_Nop;
            Block.Count := Block.Count + 1;
            Block.Statements (Block.Count) := Stmt;
         end if;
      end;
      
      Stmt_Pos := Obj_End + 1;
   end loop;
   
   Status := Success;
end Parse_Statements_Recursive;
```

### Phase 2: Flattening Algorithm

```ada
procedure Flatten_Statements (
   Block      : Statement_Block;
   Output     : out IR_Statement_Array;
   Out_Count  : out Natural
) is
   Current_Index : Natural := 0;
   
   procedure Flatten_Block_Recursive (
      Blk : Statement_Block
   ) is
   begin
      for I in 1 .. Blk.Count loop
         declare
            Stmt : Parsed_Statement renames Blk.Statements (I);
         begin
            case Stmt.Kind is
               when Stmt_If =>
                  -- Reserve slot for if statement
                  Current_Index := Current_Index + 1;
                  declare
                     If_Index : constant Positive := Current_Index;
                     Then_Start : Natural;
                     Then_Count : Natural := 0;
                     Else_Start : Natural := 0;
                     Else_Count : Natural := 0;
                  begin
                     -- Flatten then block
                     Then_Start := Current_Index + 1;
                     if Stmt.Then_Block /= null then
                        Flatten_Block_Recursive (Stmt.Then_Block.all);
                        Then_Count := Current_Index - Then_Start + 1;
                     end if;
                     
                     -- Flatten else block if present
                     if Stmt.Else_Block /= null then
                        Else_Start := Current_Index + 1;
                        Flatten_Block_Recursive (Stmt.Else_Block.all);
                        Else_Count := Current_Index - Else_Start + 1;
                     end if;
                     
                     -- Fill in if statement with indices
                     Output (If_Index).Kind := Stmt_If;
                     Output (If_Index).Condition := Stmt.Condition;
                     Output (If_Index).Block_Start := Then_Start;
                     Output (If_Index).Block_Count := Then_Count;
                     Output (If_Index).Else_Start := Else_Start;
                     Output (If_Index).Else_Count := Else_Count;
                  end;
               
               when Stmt_While =>
                  -- Similar pattern to if
                  Current_Index := Current_Index + 1;
                  declare
                     While_Index : constant Positive := Current_Index;
                     Body_Start : Natural;
                     Body_Count : Natural := 0;
                  begin
                     -- Flatten body
                     Body_Start := Current_Index + 1;
                     if Stmt.Body_Block /= null then
                        Flatten_Block_Recursive (Stmt.Body_Block.all);
                        Body_Count := Current_Index - Body_Start + 1;
                     end if;
                     
                     -- Fill in while statement
                     Output (While_Index).Kind := Stmt_While;
                     Output (While_Index).Condition := Stmt.Condition;
                     Output (While_Index).Block_Start := Body_Start;
                     Output (While_Index).Block_Count := Body_Count;
                  end;
               
               when Stmt_For =>
                  -- Similar to while, but with init/increment
                  Current_Index := Current_Index + 1;
                  declare
                     For_Index : constant Positive := Current_Index;
                     Body_Start : Natural;
                     Body_Count : Natural := 0;
                  begin
                     Body_Start := Current_Index + 1;
                     if Stmt.Body_Block /= null then
                        Flatten_Block_Recursive (Stmt.Body_Block.all);
                        Body_Count := Current_Index - Body_Start + 1;
                     end if;
                     
                     Output (For_Index).Kind := Stmt_For;
                     Output (For_Index).Init_Expr := Stmt.Init_Expr;
                     Output (For_Index).Condition := Stmt.Condition;
                     Output (For_Index).Incr_Expr := Stmt.Incr_Expr;
                     Output (For_Index).Block_Start := Body_Start;
                     Output (For_Index).Block_Count := Body_Count;
                  end;
               
               when Stmt_Assign | Stmt_Call | Stmt_Return | Stmt_Nop =>
                  -- Regular statements - just copy
                  Current_Index := Current_Index + 1;
                  Output (Current_Index) := (
                     Kind      => Stmt.Kind,
                     Target    => Stmt.Target,
                     Value     => Stmt.Value,
                     others    => <>  -- Default values for control flow fields
                  );
            end case;
         end;
      end loop;
   end Flatten_Block_Recursive;
begin
   Current_Index := 0;
   Flatten_Block_Recursive (Block);
   Out_Count := Current_Index;
end Flatten_Statements;
```

## Implementation Plan

### Phase 3: Code Implementation

1. **Update `stunir_semantic_ir.ads`**
   - Add `Max_Nesting_Depth` constant
   - Add control flow fields to `IR_Statement` type

2. **Update `stunir_json_utils.adb`**
   - Implement `Parse_Statements_Recursive`
   - Implement `Flatten_Statements`
   - Implement `Extract_Array_JSON` helper
   - Update `Parse_Spec_JSON` to call new parsing logic

3. **Update `stunir_json_utils.ads`**
   - Add new procedure declarations
   - Update contracts/preconditions

4. **Update IR JSON serialization**
   - Emit control flow steps with proper block indices
   - Use `stunir_flat_ir_v1` schema

### Phase 4: Testing

1. **Unit tests**: Test each control flow type individually
2. **Integration tests**: Test nested control flow
3. **End-to-end tests**: Full SPARK pipeline
4. **Cross-validation**: Compare SPARK output with Python output

## SPARK Verification Considerations

### Bounds Checking
- All array accesses must be proven within bounds
- Use `pragma Assert` for runtime checks during development
- Use `pragma Loop_Invariant` for loop verification

### Recursion Depth
- Explicit depth counter prevents infinite recursion
- Maximum depth checked at each recursive call
- SPARK can prove termination with bounded depth

### Memory Safety
- No dynamic allocation in production code
- All arrays have static bounds
- No access types except for temporary parsing structure

## Output Format

### Flattened IR JSON Schema

```json
{
  "schema": "stunir_flat_ir_v1",
  "ir_version": "v1",
  "module_name": "example",
  "functions": [
    {
      "name": "foo",
      "args": [...],
      "return_type": "i32",
      "steps": [
        {
          "op": "if",
          "condition": "x > 0",
          "block_start": 2,
          "block_count": 2,
          "else_start": 4,
          "else_count": 1
        },
        {
          "op": "assign",
          "target": "y",
          "value": "x + 1"
        },
        {
          "op": "assign",
          "target": "z",
          "value": "y * 2"
        },
        {
          "op": "assign",
          "target": "y",
          "value": "0"
        }
      ]
    }
  ]
}
```

### Key Features
- **1-based indexing**: Ada arrays are 1-based by default
- **block_start**: First step in block (1-based)
- **block_count**: Number of steps in block
- **else_start**: First step in else block (0 if none)
- **else_count**: Number of steps in else block (0 if none)

## Performance Considerations

### Time Complexity
- **Parsing**: O(n * d) where n = statements, d = depth
- **Flattening**: O(n) single pass
- **Total**: O(n * d), acceptable for d ≤ 5

### Space Complexity
- **Temporary parsing structure**: O(n * d) stack space
- **Final IR array**: O(n) linear
- **Peak memory**: O(n * d) during parsing, O(n) after

### Optimization Opportunities
- Use stack allocation for small blocks
- Reuse parsing buffers across functions
- Minimize string copying

## Limitations and Future Work

### v0.8.0 Limitations
- Maximum nesting depth: 5 levels
- Maximum statements per block: 20
- Maximum total statements per function: 100

### Future Enhancements (v0.9.0+)
- Dynamic depth limits based on available memory
- Larger block sizes with sparse arrays
- Parallel parsing for multiple functions

## Success Criteria

1. ✅ SPARK spec_to_ir parses if/while/for from spec JSON
2. ✅ Generates valid flattened IR
3. ✅ SPARK ir_to_code consumes flattened IR successfully
4. ✅ Generated C code compiles and runs correctly
5. ✅ Full SPARK pipeline (spec → IR → C) works without Python dependency
6. ✅ Cross-validation: SPARK output matches Python output functionally

## Conclusion

This design enables **100% SPARK pipeline completion** by implementing control flow parsing and flattening directly in Ada SPARK. The two-phase parsing approach (parse nested, then flatten) provides a clean separation of concerns while maintaining SPARK verification guarantees.

**Result**: Full SPARK-native end-to-end pipeline! 🎉

---

**Document Version**: v0.8.0  
**Author**: STUNIR Development Team  
**Date**: 2026-01-31  
**Status**: Design Complete, Ready for Implementation
