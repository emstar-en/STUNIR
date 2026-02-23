--  Assemble Spec Main - Thin CLI Wrapper
--  Assembles spec JSON from extraction data
--  Phase: 1 (Spec)
--  SPARK_Mode: Off (IO)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Command_Line;
with Ada.Text_IO;
with Assemble_Spec;
with STUNIR_Types;

procedure Assemble_Spec_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;
   
   Status : Status_Code;
begin
   if ACL.Argument_Count < 2 then
      Put_Line (Standard_Error, "Usage: assemble_spec <input_extract.json> <output_spec.json>");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;
   
   Assemble_Spec.Assemble_Spec_File
     (Input_Path  => Path_Strings.To_Bounded_String (ACL.Argument (1)),
      Output_Path => Path_Strings.To_Bounded_String (ACL.Argument (2)),
      Status      => Status);
   
   if Status = STUNIR_Types.Success then
      Put_Line ("Spec assembled successfully");
      ACL.Set_Exit_Status (ACL.Success);
   else
      Put_Line (Standard_Error, "Failed to assemble spec");
      ACL.Set_Exit_Status (ACL.Failure);
   end if;
end Assemble_Spec_Main;
