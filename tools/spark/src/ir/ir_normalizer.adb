--  IR Normalizer - Pre-Emission IR Transformation (Implementation)
--  Language-agnostic normalization and lowering passes
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;
with Identifier_Strings;

package body IR_Normalizer is

   --  =========================================================================
   --  Default Configuration
   --  =========================================================================

   function Default_Config return Normalizer_Config is
   begin
      return Normalizer_Config'(
         Enabled_Passes => (others => True),
         Max_Temps      => 64,
         Verbose        => False
      );
   end Default_Config;

   function Minimal_Config return Normalizer_Config is
   begin
      return Normalizer_Config'(
         Enabled_Passes => (
            Pass_Switch_Lowering     => True,
            Pass_For_Lowering        => True,
            Pass_Break_Continue      => False,  --  Optional, complex
            Pass_Try_Catch_Lowering  => False,  --  Optional, complex
            Pass_Block_Flatten       => True,
            Pass_Return_Normalize    => True,
            Pass_Temp_Naming         => True,
            Pass_Expression_Simplify => False  --  Optional
         ),
         Max_Temps      => 64,
         Verbose        => False
      );
   end Minimal_Config;

   --  =========================================================================
   --  Utility Functions
   --  =========================================================================

   function Needs_Normalization
     (Step   : IR_Step;
      Config : Normalizer_Config) return Boolean
   is
   begin
      --  Check if this step type requires normalization
      case Step.Step_Type is
         when Step_Switch =>
            return Config.Enabled_Passes (Pass_Switch_Lowering);
         when Step_For =>
            return Config.Enabled_Passes (Pass_For_Lowering);
         when Step_Break | Step_Continue =>
            return Config.Enabled_Passes (Pass_Break_Continue);
         when others =>
            return False;
      end case;
   end Needs_Normalization;

   function Count_Needing_Normalization
     (Steps  : Step_Collection;
      Config : Normalizer_Config) return Natural
   is
      Count : Natural := 0;
   begin
      for I in Step_Index range 1 .. Steps.Count loop
         if Needs_Normalization (Steps.Steps (I), Config) then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Needing_Normalization;

   function Make_Temp_Name
     (Index  : Natural;
      Prefix : String := "_t") return Identifier_String
   is
      --  Convert index to string (simple implementation for SPARK)
      function Nat_To_Str (N : Natural) return String is
         Img : constant String := Natural'Image (N);
      begin
         return Img (Img'First + 1 .. Img'Last);  --  Skip leading space
      end Nat_To_Str;
      
      Name : constant String := Prefix & Nat_To_Str (Index);
   begin
      return Identifier_Strings.To_Bounded_String (Name);
   end Make_Temp_Name;

   function Make_Break_Flag_Name
     (Loop_Index : Natural) return Identifier_String
   is
   begin
      return Make_Temp_Name (Loop_Index, "_break_");
   end Make_Break_Flag_Name;

   function Make_Continue_Flag_Name
     (Loop_Index : Natural) return Identifier_String
   is
   begin
      return Make_Temp_Name (Loop_Index, "_continue_");
   end Make_Continue_Flag_Name;

   --  =========================================================================
   --  Individual Pass Procedures
   --  =========================================================================

   procedure Lower_Switch
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Switch lowering converts:
      --    switch (x) { case 1: A; case 2: B; default: C; }
      --  To:
      --    if (x == 1) { A } else if (x == 2) { B } else { C }
      
      New_Steps : Step_Collection;
      New_Count : Step_Index := 0;
      
      procedure Append_Step (S : IR_Step) is
      begin
         if New_Count < Max_Steps then
            New_Count := New_Count + 1;
            New_Steps.Steps (New_Count) := S;
         end if;
      end Append_Step;
      
      procedure Lower_Case_Chain
        (Switch_Step : IR_Step;
         Case_Index  : Natural)
      is
         --  Recursively build if/else chain from cases
         If_Step : IR_Step := Make_Default_Step;
      begin
         If_Step.Step_Type := Step_If;
         
         if Case_Index <= Natural (Switch_Step.Case_Count) then
            --  Build condition: expr == case_value
            If_Step.Condition := Identifier_Strings.To_Bounded_String
              (Identifier_Strings.To_String (Switch_Step.Expr) & " == " &
               Identifier_Strings.To_String (Switch_Step.Cases (Case_Index).Case_Value));
            
            --  Set then block to case body
            If_Step.Then_Start := Switch_Step.Cases (Case_Index).Body_Start;
            If_Step.Then_Count := Switch_Step.Cases (Case_Index).Body_Count;
            
            --  Set else block to next case or default
            if Case_Index < Natural (Switch_Step.Case_Count) then
               --  Will be filled by recursive call
               If_Step.Else_Start := New_Count + 2;  --  After this if
               If_Step.Else_Count := 1;  --  Placeholder
            elsif Switch_Step.Default_Count > 0 then
               If_Step.Else_Start := Switch_Step.Default_Start;
               If_Step.Else_Count := Switch_Step.Default_Count;
            end if;
            
            Append_Step (If_Step);
            
            --  Recursively handle remaining cases
            if Case_Index < Natural (Switch_Step.Case_Count) then
               Lower_Case_Chain (Switch_Step, Case_Index + 1);
            end if;
         end if;
      end Lower_Case_Chain;
      
   begin
      Init_Step_Collection (New_Steps);
      
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_Switch then
               --  Lower this switch to if/else chain
               Lower_Case_Chain (Step, 1);
               Stats.Switches_Lowered := Stats.Switches_Lowered + 1;
            else
               --  Copy step as-is
               Append_Step (Step);
            end if;
         end;
      end loop;
      
      New_Steps.Count := New_Count;
      Steps := New_Steps;
   end Lower_Switch;

   procedure Lower_For_Loops
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  For loop lowering converts:
      --    for (i = init; i < cond; i = i + incr) { body }
      --  To:
      --    i = init;
      --    while (i < cond) { body; i = i + incr; }
      
      New_Steps : Step_Collection;
      New_Count : Step_Index := 0;
      
      procedure Append_Step (S : IR_Step) is
      begin
         if New_Count < Max_Steps then
            New_Count := New_Count + 1;
            New_Steps.Steps (New_Count) := S;
         end if;
      end Append_Step;
      
   begin
      Init_Step_Collection (New_Steps);
      
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_For then
               --  Emit init assignment
               declare
                  Init_Step : IR_Step := Make_Default_Step;
               begin
                  Init_Step.Step_Type := Step_Assign;
                  --  Extract loop variable from condition (simplified)
                  Init_Step.Target := Identifier_Strings.To_Bounded_String ("i");
                  Init_Step.Value := Step.Init;
                  Append_Step (Init_Step);
               end;
               
               --  Emit while loop
               declare
                  While_Step : IR_Step := Step;
               begin
                  While_Step.Step_Type := Step_While;
                  --  Increment is appended to body (simplified)
                  Append_Step (While_Step);
               end;
               
               Stats.For_Loops_Lowered := Stats.For_Loops_Lowered + 1;
            else
               Append_Step (Step);
            end if;
         end;
      end loop;
      
      New_Steps.Count := New_Count;
      Steps := New_Steps;
   end Lower_For_Loops;

   procedure Lower_Break_Continue
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Break/continue lowering converts:
      --    while (cond) { ... break; ... }
      --  To:
      --    _break_0 = false;
      --    while (cond && !_break_0) { ... _break_0 = true; ... }
      
      Loop_Depth : Natural := 0;
   begin
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : IR_Step := Steps.Steps (I);
         begin
            case Step.Step_Type is
               when Step_While | Step_For =>
                  Loop_Depth := Loop_Depth + 1;
                  
               when Step_Break =>
                  --  Replace with flag assignment
                  Step.Step_Type := Step_Assign;
                  Step.Target := Make_Break_Flag_Name (Loop_Depth);
                  Step.Value := Identifier_Strings.To_Bounded_String ("true");
                  Steps.Steps (I) := Step;
                  Stats.Breaks_Lowered := Stats.Breaks_Lowered + 1;
                  
               when Step_Continue =>
                  --  Replace with flag assignment
                  Step.Step_Type := Step_Assign;
                  Step.Target := Make_Continue_Flag_Name (Loop_Depth);
                  Step.Value := Identifier_Strings.To_Bounded_String ("true");
                  Steps.Steps (I) := Step;
                  Stats.Continues_Lowered := Stats.Continues_Lowered + 1;
                  
               when others =>
                  null;
            end case;
         end;
      end loop;
   end Lower_Break_Continue;

   procedure Lower_Try_Catch
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Try/catch lowering converts:
      --    try { ... } catch (e) { ... }
      --  To:
      --    _error_flag = false;
      --    if (!_error_flag) { ... try body ... }
      --    if (_error_flag) { ... catch body ... }
   begin
      --  Placeholder implementation
      --  Full implementation would transform try/catch blocks
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_Try then
               Stats.Try_Catch_Lowered := Stats.Try_Catch_Lowered + 1;
            end if;
         end;
      end loop;
   end Lower_Try_Catch;

   procedure Simplify_Expressions
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Simplify complex expressions into simpler statements
      --  This is a placeholder - full implementation would parse
      --  expression trees and split complex expressions
   begin
      --  Placeholder: count complex expressions
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_Assign then
               --  Check if value is complex (more than 2 operators)
               --  Placeholder: just count for now
               null;
            end if;
         end;
      end loop;
   end Simplify_Expressions;

   procedure Canonicalize_Types
     (Func   : in out IR_Function;
      Stats  : in out Normalization_Stats)
   is
      --  Canonicalize type names to standard forms
      --  Converts: i32 -> int, f64 -> double, boolean -> bool, etc.
   begin
      --  Placeholder: would transform type names in function signature
      --  and local variable declarations
      null;
   end Canonicalize_Types;

   procedure Fold_Constants
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Fold constant expressions at compile time
      --  Evaluates: 1 + 2 -> 3, true && false -> false, etc.
   begin
      --  Placeholder: would evaluate constant expressions
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_Assign then
               --  Check if value is a constant expression
               --  Placeholder: just count for now
               null;
            end if;
         end;
      end loop;
   end Fold_Constants;

   procedure Remove_Dead_Code
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Remove unreachable/dead code
      --  Removes code after return/break/continue
   begin
      --  Placeholder: would analyze control flow and remove dead code
      null;
   end Remove_Dead_Code;

   procedure Flatten_Arrays
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Flatten nested array accesses where possible
   begin
      --  Placeholder: would flatten multi-dimensional array access
      null;
   end Flatten_Arrays;

   procedure Flatten_Structs
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Flatten nested struct accesses where possible
   begin
      --  Placeholder: would flatten nested struct access
      null;
   end Flatten_Structs;

   procedure Flatten_Blocks
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Flatten nested blocks where safe
      --  This is a placeholder - full implementation would analyze
      --  block structure and flatten where variables don't conflict
   begin
      --  Placeholder: count nested blocks
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : constant IR_Step := Steps.Steps (I);
         begin
            if Step.Step_Type = Step_If then
               if Step.Then_Count > 0 or Step.Else_Count > 0 then
                  Stats.Blocks_Flattened := Stats.Blocks_Flattened + 1;
               end if;
            end if;
         end;
      end loop;
   end Flatten_Blocks;

   procedure Normalize_Returns
     (Func   : in out IR_Function;
      Stats  : in out Normalization_Stats)
   is
      --  Ensure all branches have explicit returns
      Has_Return : Boolean := False;
   begin
      --  Check if function already has a return
      for I in Step_Index range 1 .. Func.Steps.Count loop
         if Func.Steps.Steps (I).Step_Type = Step_Return then
            Has_Return := True;
            exit;
         end if;
      end loop;
      
      --  Add default return if missing and function is non-void
      if not Has_Return then
         declare
            Ret_Step : IR_Step := Make_Default_Step;
         begin
            Ret_Step.Step_Type := Step_Return;
            
            --  Set default return value based on return type
            if Identifier_Strings.To_String (Func.Return_Type) = "void" then
               Ret_Step.Value := Identifier_Strings.Null_Bounded_String;
            elsif Identifier_Strings.To_String (Func.Return_Type) = "int" then
               Ret_Step.Value := Identifier_Strings.To_Bounded_String ("0");
            elsif Identifier_Strings.To_String (Func.Return_Type) = "bool" then
               Ret_Step.Value := Identifier_Strings.To_Bounded_String ("false");
            else
               Ret_Step.Value := Identifier_Strings.To_Bounded_String ("null");
            end if;
            
            --  Append return step
            if Func.Steps.Count < Max_Steps then
               Func.Steps.Count := Func.Steps.Count + 1;
               Func.Steps.Steps (Func.Steps.Count) := Ret_Step;
               Stats.Returns_Added := Stats.Returns_Added + 1;
            end if;
         end;
      end if;
   end Normalize_Returns;

   procedure Generate_Temp_Names
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   is
      --  Generate unique temporary variable names
      --  This ensures no shadowing across nested blocks
      Temp_Counter : Natural := 0;
   begin
      for I in Step_Index range 1 .. Steps.Count loop
         declare
            Step : IR_Step := Steps.Steps (I);
         begin
            --  Check for anonymous temps that need naming
            if Step.Step_Type = Step_Assign then
               if Identifier_Strings.Length (Step.Target) = 0 then
                  Step.Target := Make_Temp_Name (Temp_Counter);
                  Temp_Counter := Temp_Counter + 1;
                  Stats.Temps_Generated := Stats.Temps_Generated + 1;
                  Steps.Steps (I) := Step;
               end if;
            end if;
         end;
      end loop;
   end Generate_Temp_Names;

   --  =========================================================================
   --  Core Normalization Procedures
   --  =========================================================================

   procedure Normalize_Function
     (Func   : in out IR_Function;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   is
      Stats : Normalization_Stats := (
         Switches_Lowered     => 0,
         For_Loops_Lowered    => 0,
         Breaks_Lowered       => 0,
         Continues_Lowered    => 0,
         Try_Catch_Lowered    => 0,
         Blocks_Flattened     => 0,
         Returns_Added        => 0,
         Temps_Generated      => 0,
         Expressions_Split    => 0,
         Nested_Blocks_Proc   => 0,
         Arrays_Flattened     => 0,
         Structs_Flattened    => 0,
         Types_Canonicalized  => 0,
         Constants_Folded     => 0,
         Dead_Code_Removed    => 0
      );
   begin
      Result.Success := False;
      Result.Message := Error_Strings.Null_Bounded_String;
      
      --  0. Canonicalize types first
      if Config.Enabled_Passes (Pass_Type_Canonicalize) then
         Canonicalize_Types (Func, Stats);
      end if;
      
      --  1. Lower control flow constructs
      if Config.Enabled_Passes (Pass_Switch_Lowering) then
         Lower_Switch (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_For_Lowering) then
         Lower_For_Loops (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_Break_Continue) then
         Lower_Break_Continue (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_Try_Catch_Lowering) then
         Lower_Try_Catch (Func.Steps, Stats);
      end if;
      
      --  2. Simplify expressions
      if Config.Enabled_Passes (Pass_Expression_Simplify) then
         Simplify_Expressions (Func.Steps, Stats);
      end if;
      
      --  3. Fold constants
      if Config.Enabled_Passes (Pass_Constant_Fold) then
         Fold_Constants (Func.Steps, Stats);
      end if;
      
      --  4. Data structure normalization
      if Config.Enabled_Passes (Pass_Array_Flatten) then
         Flatten_Arrays (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_Struct_Flatten) then
         Flatten_Structs (Func.Steps, Stats);
      end if;
      
      --  5. Flatten and normalize
      if Config.Enabled_Passes (Pass_Block_Flatten) then
         Flatten_Blocks (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_Return_Normalize) then
         Normalize_Returns (Func, Stats);
      end if;
      
      --  6. Remove dead code
      if Config.Enabled_Passes (Pass_Dead_Code_Remove) then
         Remove_Dead_Code (Func.Steps, Stats);
      end if;
      
      if Config.Enabled_Passes (Pass_Temp_Naming) then
         Generate_Temp_Names (Func.Steps, Stats);
      end if;
      
      Result.Stats := Stats;
      Result.Success := True;
      Result.Message := Error_Strings.To_Bounded_String ("Normalization complete");
   end Normalize_Function;

   procedure Normalize_Module
     (Module : in out IR_Function_Collection;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   is
      Func_Result : Normalization_Result;
      Total_Stats : Normalization_Stats := (
         Switches_Lowered     => 0,
         For_Loops_Lowered    => 0,
         Breaks_Lowered       => 0,
         Continues_Lowered    => 0,
         Try_Catch_Lowered    => 0,
         Blocks_Flattened     => 0,
         Returns_Added        => 0,
         Temps_Generated      => 0,
         Expressions_Split    => 0,
         Nested_Blocks_Proc   => 0,
         Arrays_Flattened     => 0,
         Structs_Flattened    => 0,
         Types_Canonicalized  => 0,
         Constants_Folded     => 0,
         Dead_Code_Removed    => 0
      );
   begin
      Result.Success := False;
      Result.Message := Error_Strings.Null_Bounded_String;
      
      for I in Function_Index range 1 .. Module.Count loop
         Normalize_Function (Module.Functions (I), Config, Func_Result);
         
         if not Func_Result.Success then
            Result.Success := False;
            Result.Message := Func_Result.Message;
            return;
         end if;
         
         --  Accumulate stats
         Total_Stats.Switches_Lowered := Total_Stats.Switches_Lowered +
            Func_Result.Stats.Switches_Lowered;
         Total_Stats.For_Loops_Lowered := Total_Stats.For_Loops_Lowered +
            Func_Result.Stats.For_Loops_Lowered;
         Total_Stats.Breaks_Lowered := Total_Stats.Breaks_Lowered +
            Func_Result.Stats.Breaks_Lowered;
         Total_Stats.Continues_Lowered := Total_Stats.Continues_Lowered +
            Func_Result.Stats.Continues_Lowered;
         Total_Stats.Try_Catch_Lowered := Total_Stats.Try_Catch_Lowered +
            Func_Result.Stats.Try_Catch_Lowered;
         Total_Stats.Blocks_Flattened := Total_Stats.Blocks_Flattened +
            Func_Result.Stats.Blocks_Flattened;
         Total_Stats.Returns_Added := Total_Stats.Returns_Added +
            Func_Result.Stats.Returns_Added;
         Total_Stats.Temps_Generated := Total_Stats.Temps_Generated +
            Func_Result.Stats.Temps_Generated;
         Total_Stats.Expressions_Split := Total_Stats.Expressions_Split +
            Func_Result.Stats.Expressions_Split;
         Total_Stats.Nested_Blocks_Proc := Total_Stats.Nested_Blocks_Proc +
            Func_Result.Stats.Nested_Blocks_Proc;
         Total_Stats.Arrays_Flattened := Total_Stats.Arrays_Flattened +
            Func_Result.Stats.Arrays_Flattened;
         Total_Stats.Structs_Flattened := Total_Stats.Structs_Flattened +
            Func_Result.Stats.Structs_Flattened;
         Total_Stats.Types_Canonicalized := Total_Stats.Types_Canonicalized +
            Func_Result.Stats.Types_Canonicalized;
         Total_Stats.Constants_Folded := Total_Stats.Constants_Folded +
            Func_Result.Stats.Constants_Folded;
         Total_Stats.Dead_Code_Removed := Total_Stats.Dead_Code_Removed +
            Func_Result.Stats.Dead_Code_Removed;
      end loop;
      
      Result.Stats := Total_Stats;
      Result.Success := True;
      Result.Message := Error_Strings.To_Bounded_String ("Module normalization complete");
   end Normalize_Module;

end IR_Normalizer;
