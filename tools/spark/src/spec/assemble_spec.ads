--  Assemble Spec Micro-Tool
--  Assembles spec JSON from extraction data
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;
with Extract_Parse;
with Spec_Parse;

package Assemble_Spec is

   --  Assemble spec from extraction data
   procedure Assemble_From_Extract
     (Extract : in     Extract_Parse.Extract_Data;
      Spec    :    out Spec_Parse.Spec_Data;
      Status  :    out Status_Code);

   --  Assemble spec JSON file from extraction JSON file
   procedure Assemble_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Status      :    out Status_Code);

end Assemble_Spec;
