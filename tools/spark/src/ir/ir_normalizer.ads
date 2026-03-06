--  IR Normalizer - Pre-Emission IR Transformation
--  Language-agnostic normalization and lowering passes
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides normalization passes that run before emission
--  to simplify the IR into a minimal core that emitters can easily render.
--
--  Normalization passes:
--  1. Switch lowering - Convert switch to nested if/else
--  2. For loop lowering - Convert for to while with explicit init/increment
--  3. Break/continue lowering - Convert to explicit loop state flags
--  4. Block flattening - Flatten nested blocks where safe
--  5. Return normalization - Ensure all branches have explicit returns
--  6. Temp naming - Generate unique temporary variable names

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package IR_Normalizer is

   --  =========================================================================
   --  Normalization Configuration
   --  =========================================================================

   type Normalization_Pass is (
      Pass_Switch_Lowering,      --  Convert switch to if/else
      Pass_For_Lowering,         --  Convert for to while
      Pass_Break_Continue,       --  Lower break/continue to flags
      Pass_Block_Flatten,        --  Flatten nested blocks
      Pass_Return_Normalize,     --  Ensure explicit returns
      Pass_Temp_Naming,          --  Unique temp variable names
      Pass_Expression_Simplify   --  Simplify complex expressions
   );

   type Pass_Enabled is array (Normalization_Pass) of Boolean;

   type Normalizer_Config is record
      Enabled_Passes : Pass_Enabled := (others => True);
      Max_Temps      : Natural := 64;  --  Max temp variables to generate
      Verbose        : Boolean := False;
   end record;

   --  =========================================================================
   --  Normalization Results
   --  =========================================================================

   type Normalization_Stats is record
      Switches_Lowered   : Natural;
      For_Loops_Lowered  : Natural;
      Breaks_Lowered     : Natural;
      Continues_Lowered  : Natural;
      Blocks_Flattened   : Natural;
      Returns_Added      : Natural;
      Temps_Generated    : Natural;
      Expressions_Split  : Natural;
   end record;

   type Normalization_Result is record
      Success      : Boolean;
      Stats        : Normalization_Stats;
      Message      : Error_String;
   end record;

   --  =========================================================================
   --  Core Normalization Procedures
   --  =========================================================================

   --  Normalize a single IR function in place
   procedure Normalize_Function
     (Func   : in out IR_Function;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   with
      Pre  => Identifier_Strings.Length (Func.Name) > 0,
      Post => (if Result.Success then Func.Steps.Count <= Max_Steps);

   --  Normalize all functions in an IR module
   procedure Normalize_Module
     (Module : in out IR_Function_Collection;
      Config : in     Normalizer_Config;
      Result :    out Normalization_Result)
   with
      Pre  => Module.Count <= Max_Functions,
      Post => (if Result.Success then 
                 (for all I in 1 .. Module.Count =>
                    Module.Functions (I).Steps.Count <= Max_Steps));

   --  =========================================================================
   --  Individual Pass Procedures
   --  =========================================================================

   --  Lower switch statements to nested if/else chains
   procedure Lower_Switch
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Steps.Count <= Max_Steps,
      Post => Steps.Count <= Max_Steps;

   --  Lower for loops to while loops with explicit init/increment
   procedure Lower_For_Loops
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Steps.Count <= Max_Steps,
      Post => Steps.Count <= Max_Steps;

   --  Lower break/continue to explicit loop state flags
   --  This converts:
   --    while (cond) { ... break; ... }
   --  To:
   --    bool _break_0 = false;
   --    while (cond && !_break_0) { ... _break_0 = true; ... }
   procedure Lower_Break_Continue
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Steps.Count <= Max_Steps,
      Post => Steps.Count <= Max_Steps;

   --  Flatten nested blocks where safe
   procedure Flatten_Blocks
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Steps.Count <= Max_Steps,
      Post => Steps.Count <= Max_Steps;

   --  Ensure all branches have explicit returns
   procedure Normalize_Returns
     (Func   : in out IR_Function;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Identifier_Strings.Length (Func.Name) > 0,
      Post => Func.Steps.Count <= Max_Steps;

   --  Generate unique temporary variable names
   procedure Generate_Temp_Names
     (Steps  : in out Step_Collection;
      Stats  : in out Normalization_Stats)
   with
      Pre  => Steps.Count <= Max_Steps,
      Post => Steps.Count <= Max_Steps;

   --  =========================================================================
   --  Utility Functions
   --  =========================================================================

   --  Check if a step type requires normalization
   function Needs_Normalization
     (Step   : IR_Step;
      Config : Normalizer_Config) return Boolean
   with Global => null;

   --  Count steps that need normalization
   function Count_Needing_Normalization
     (Steps  : Step_Collection;
      Config : Normalizer_Config) return Natural
   with Global => null;

   --  Generate a unique temp variable name
   function Make_Temp_Name
     (Index  : Natural;
      Prefix : String := "_t") return Identifier_String
   with
      Pre  => Index < 1000,  --  Reasonable limit
      Post => Identifier_Strings.Length (Make_Temp_Name'Result) > 0;

   --  Generate a break flag name for a loop
   function Make_Break_Flag_Name
     (Loop_Index : Natural) return Identifier_String
   with
      Pre  => Loop_Index < 100,
      Post => Identifier_Strings.Length (Make_Break_Flag_Name'Result) > 0;

   --  Generate a continue flag name for a loop
   function Make_Continue_Flag_Name
     (Loop_Index : Natural) return Identifier_String
   with
      Pre  => Loop_Index < 100,
      Post => Identifier_Strings.Length (Make_Continue_Flag_Name'Result) > 0;

   --  =========================================================================
   --  Default Configuration
   --  =========================================================================

   --  Default configuration with all passes enabled
   function Default_Config return Normalizer_Config
   with Global => null;

   --  Minimal configuration (only essential passes)
   function Minimal_Config return Normalizer_Config
   with Global => null;

end IR_Normalizer;
