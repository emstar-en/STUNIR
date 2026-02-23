--  source_extract_main - Minimal source -> extraction.json (C/C++)
--  Phase 0: Source extraction
--  SPARK_Mode: Off (file I/O + regex-like parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
with Source_Extract;

procedure Source_Extract_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;
   use Source_Extract;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Module_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
   Language    : Identifier_String := Identifier_Strings.To_Bounded_String ("c");

   Exit_Failure : constant := 1;

begin
   if ACL.Argument_Count < 2 then
      Put_Line ("Usage: source_extract_main -i <source> -o <extraction.json> [-m <module>] [--lang <lang>]");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   declare
      I : Positive := 1;
   begin
      while I <= ACL.Argument_Count loop
         declare
            Arg : constant String := ACL.Argument (I);
         begin
            if Arg = "-i" and then I < ACL.Argument_Count then
               I := I + 1;
               Input_Path := Path_Strings.To_Bounded_String (ACL.Argument (I));
            elsif Arg = "-o" and then I < ACL.Argument_Count then
               I := I + 1;
               Output_Path := Path_Strings.To_Bounded_String (ACL.Argument (I));
            elsif Arg = "-m" and then I < ACL.Argument_Count then
               I := I + 1;
               Module_Name := Identifier_Strings.To_Bounded_String (ACL.Argument (I));
            elsif Arg = "--lang" and then I < ACL.Argument_Count then
               I := I + 1;
               Language := Identifier_Strings.To_Bounded_String (ACL.Argument (I));
            else
               Put_Line ("Error: Unknown argument: " & Arg);
               ACL.Set_Exit_Status (ACL.Failure);
               return;
            end if;
            I := I + 1;
         end;
      end loop;
   end;

   if Path_Strings.Length (Input_Path) = 0 or else Path_Strings.Length (Output_Path) = 0 then
      Put_Line ("Error: -i and -o are required");
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   if Identifier_Strings.Length (Module_Name) = 0 then
      Module_Name := Identifier_Strings.To_Bounded_String ("module");
   end if;

   declare
      Status : Status_Code;
   begin
      Extract_File
        (Input_Path  => Input_Path,
         Output_Path => Output_Path,
         Module_Name => Module_Name,
         Language    => Language,
         Status      => Status);

      if Status = Success then
         ACL.Set_Exit_Status (ACL.Success);
      else
         Put_Line ("Error: " & Status_Code_Image (Status));
         ACL.Set_Exit_Status (ACL.Failure);
      end if;
   end;
exception
   when others =>
      Put_Line ("Error: source_extract_main failed");
      ACL.Set_Exit_Status (Exit_Failure);
end Source_Extract_Main;
