--  IR Converter Main Program
--  Command-line entry point for spec.json to ir.json conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);  --  Command-line parsing not in SPARK

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
use STUNIR_Types;
with Spec_To_IR;

procedure IR_Converter_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;

   procedure Print_Usage is
   begin
      Put_Line ("Usage: ir_converter <input_spec.json> <output_ir.json>");
      Put_Line ("  input_spec.json    Input spec.json file path");
      Put_Line ("  output_ir.json     Output ir.json file path");
      Put_Line ("  -h                 Show this help message");
   end Print_Usage;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Status      : Status_Code;

begin
   --  Parse command-line arguments
   if ACL.Argument_Count < 2 then
      Print_Usage;
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   declare
      Arg : constant String := ACL.Argument (1);
   begin
      if Arg = "-h" or Arg = "--help" then
         Print_Usage;
         ACL.Set_Exit_Status (ACL.Success);
         return;
      end if;
   end;

   Input_Path := Path_Strings.To_Bounded_String (ACL.Argument (1));
   Output_Path := Path_Strings.To_Bounded_String (ACL.Argument (2));

   --  Process the spec file
   Put_Line ("Converting spec to IR...");
   Put_Line ("  Input:  " & Path_Strings.To_String (Input_Path));
   Put_Line ("  Output: " & Path_Strings.To_String (Output_Path));

   Spec_To_IR.Convert_Spec_File (Input_Path, Output_Path, Status);

   if Status = STUNIR_Types.Success then
      Put_Line ("Conversion successful.");
      ACL.Set_Exit_Status (ACL.Success);
   else
      Put_Line ("Error: Conversion failed with status " & Status_Code'Image (Status));
      ACL.Set_Exit_Status (ACL.Failure);
   end if;

end IR_Converter_Main;