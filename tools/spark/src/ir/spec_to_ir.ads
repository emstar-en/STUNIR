--  Spec to IR Micro-Tool
--  Converts spec JSON to IR JSON
--  Phase: 2 (IR)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;
with Spec_Parse;

package Spec_To_IR is

   --  Convert Spec_Data to IR_Data
   procedure Convert_Spec_To_IR
     (Spec   : in     Spec_Parse.Spec_Data;
      IR     :    out IR_Data;
      Status :    out Status_Code);

   --  Convert spec JSON file to IR JSON file
   procedure Convert_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Status      :    out Status_Code);

end Spec_To_IR;
