--  STUNIR Spec Assembler Main Program
--  Command-line entry point for extraction.json to spec.json conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
with Spec_Assembler;

procedure Spec_Assembler_Main is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;

   Input_Path  : Path_String;
   Output_Path : Path_String;
   Module_Name : Identifier_String;
   Status      : Status_Code;
begin
   --  Check command line arguments
   if Argument_Count < 4 then
      Put_Line ("Usage: spec_assembler -i <input.json> -o <output.json> [-m <module_name>]");
      Put_Line ("  -i <input.json>    Path to extraction.json input file");
      Put_Line ("  -o <output.json>   Path to spec.json output file");
      Put_Line ("  -m <module_name>   Module name (optional, defaults to 'module')");
      Set_Exit_Status (Failure);
      return;
   end if;

   --  Parse arguments
   declare
      I : Positive := 1;
   begin
      while I <= Argument_Count loop
         declare
            Arg : constant String := Argument (I);
         begin
            if Arg = "-i" and I < Argument_Count then
               I := I + 1;
               Input_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "-o" and I < Argument_Count then
               I := I + 1;
               Output_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "-m" and I < Argument_Count then
               I := I + 1;
               Module_Name := Identifier_Strings.To_Bounded_String (Argument (I));
            else
               Put_Line ("Error: Unknown argument: " & Arg);
               Set_Exit_Status (Failure);
               return;
            end if;
            I := I + 1;
         end;
      end loop;
   end;

   --  Validate required arguments
   if Path_Strings.Length (Input_Path) = 0 then
      Put_Line ("Error: Input path is required (-i)");
      Set_Exit_Status (Failure);
      return;
   end if;

   if Path_Strings.Length (Output_Path) = 0 then
      Put_Line ("Error: Output path is required (-o)");
      Set_Exit_Status (Failure);
      return;
   end if;

   --  Set default module name if not provided
   if Identifier_Strings.Length (Module_Name) = 0 then
      Module_Name := Identifier_Strings.To_Bounded_String ("module");
   end if;

   --  Process the extraction file
   Spec_Assembler.Process_Extraction_File
     (Input_Path  => Input_Path,
      Output_Path => Output_Path,
      Module_Name => Module_Name,
      Status      => Status);

   --  Handle result
   if Is_Success (Status) then
      Put_Line ("Successfully converted extraction.json to spec.json");
      Put_Line ("Output: " & Path_Strings.To_String (Output_Path));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   else
      Put_Line ("Error: " & Status_Code_Image (Status));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;

end Spec_Assembler_Main;
