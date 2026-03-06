--  Emit Target Micro-Tool
--  Emits target language code from IR
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Emit_Target is

   --  Emission mode for functions (from artifact normalization)
   type Emission_Mode is (
      Emit_Source,     --  Generate source code
      Emit_Binary,     --  Embed pre-compiled binary
      Emit_Hybrid      --  Source with binary fallback
   );

   --  ========================================================================
   --  Emission Policy (Stub Hints vs Best Effort)
   --  ========================================================================

   --  Check if function requests stub hints (default) or best-effort code
   function Should_Emit_Stub_Hints (Func : IR_Function) return Boolean;

   --  Check if function requests best-effort code generation
   function Should_Emit_Best_Effort (Func : IR_Function) return Boolean;

   --  ========================================================================
   --  Shared Emitter Helpers (v0.9.1+)
   --  ========================================================================

   --  Get zero-value default for a type (for empty body placeholders)
   function Get_Zero_Value (Type_Name : String) return String;

   --  Escape string literal for target language
   function Escape_String_Literal
     (Text   : String;
      Target : Target_Language) return String;

   --  Generate JSONPath pointer for a step
   function Make_Step_Pointer
     (Func_Idx : Natural;
      Step_Idx : Natural) return String;

   --  Generate stub hint comment for a step
   function Make_Stub_Hint
     (Func_Idx  : Natural;
      Step_Idx  : Natural;
      Op_Name   : String;
      Key_Fields : String;
      Target    : Target_Language) return String;

   --  Generate conditional stub hint (only if emission mode is stub_hints)
   function Make_Conditional_Stub_Hint
     (Func      : IR_Function;
      Func_Idx  : Natural;
      Step_Idx  : Natural;
      Op_Name   : String;
      Key_Fields : String;
      Target    : Target_Language) return String;

   --  Check if a type is a primitive numeric type
   function Is_Numeric_Type (Type_Name : String) return Boolean;

   --  Check if a type is a boolean type
   function Is_Boolean_Type (Type_Name : String) return Boolean;

   --  Check if a type is void
   function Is_Void_Type (Type_Name : String) return Boolean;

   --  Get comment prefix for target language
   function Get_Comment_Prefix (Target : Target_Language) return String;

   --  Get block start/end delimiters for target language
   function Get_Block_Start (Target : Target_Language) return String;
   function Get_Block_End (Target : Target_Language) return String;

   --  Emit code for a single target language
   procedure Emit_Single_Target
     (IR       : in     IR_Data;
      Target   : in     Target_Language;
      Code     :    out Code_String;
      Status   :    out Status_Code);

   --  Emit code from IR file to output file
   procedure Emit_Target_File
     (Input_Path  : in     Path_String;
      Target      : in     Target_Language;
      Output_Path : in     Path_String;
      Status      :    out Status_Code);

   --  Get file extension for target
   function Get_Target_Extension (Target : Target_Language) return String;

   --  ========================================================================
   --  Artifact-Aware Emission (v0.9.0+)
   --  ========================================================================

   --  Check if a function should emit source or use pre-compiled binary
   function Get_Function_Emission_Mode
     (IR        : IR_Data;
      Func_Name : Identifier_String;
      Target    : Target_Language) return Emission_Mode;

   --  Check if a function has a matching GPU binary
   function Has_GPU_Binary
     (IR        : IR_Data;
      Func_Name : Identifier_String) return Boolean;

   --  Get the binary path for a function's GPU binary (if available)
   function Get_GPU_Binary_Path
     (IR        : IR_Data;
      Func_Name : Identifier_String) return Path_String;

   --  Get the binary format for a function's GPU binary (if available)
   function Get_GPU_Binary_Format
     (IR        : IR_Data;
      Func_Name : Identifier_String) return GPU_Binary_Format;

end Emit_Target;
