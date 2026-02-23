--  STUNIR Spec Assembler Main Program
--  Command-line entry point for extraction.json to spec.json conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
with Assemble_Spec;

procedure Spec_Assembler_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;

   Input_Path  : Path_String;
   Output_Path : Path_String;
   Status      : Status_Code;
begin
   --  Check command line arguments
   if ACL.Argument_Count < 2 then
      Put_Line ("Usage: spec_assembler <input_extract.json> <output_spec.json>");
      Put_Line ("  input_extract.json    Path to extraction.json input file");
      Put_Line ("  output_spec.json      Path to spec.json output file");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   Input_Path := Path_Strings.To_Bounded_String (ACL.Argument (1));
   Output_Path := Path_Strings.To_Bounded_String (ACL.Argument (2));

   --  Process the extraction file
   Assemble_Spec.Assemble_Spec_File
     (Input_Path  => Input_Path,
      Output_Path => Output_Path,
      Status      => Status);

   --  Handle result
   if Status = STUNIR_Types.Success then
      Put_Line ("Successfully converted extraction.json to spec.json");
      Put_Line ("Output: " & Path_Strings.To_String (Output_Path));
      ACL.Set_Exit_Status (ACL.Success);
   else
      Put_Line ("Error: " & Status_Code_Image (Status));
      ACL.Set_Exit_Status (ACL.Failure);
   end if;

end Spec_Assembler_Main;
