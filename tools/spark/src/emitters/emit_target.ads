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
