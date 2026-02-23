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

end Emit_Target;
