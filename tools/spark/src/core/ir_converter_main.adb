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
with IR_Converter;

procedure IR_Converter_Main is

   procedure Print_Usage is
   begin
      Ada.Text_IO.Put_Line ("Usage: ir_converter -i <input> -o <output> -m <module>");
      Ada.Text_IO.Put_Line ("  -i <input>    Input spec.json file path");
      Ada.Text_IO.Put_Line ("  -o <output>   Output ir.json file path");
      Ada.Text_IO.Put_Line ("  -m <module>   Module name for IR");
      Ada.Text_IO.Put_Line ("  -h            Show this help message");
   end Print_Usage;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Module_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
   Status      : Status_Code;
   Arg_Index   : Positive := 1;

begin
   --  Parse command-line arguments
   while Arg_Index <= Ada.Command_Line.Argument_Count loop
      declare
         Arg : constant String := Ada.Command_Line.Argument (Arg_Index);
      begin
         if Arg = "-h" or Arg = "--help" then
            Print_Usage;
            Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
            return;
         elsif Arg = "-i" then
            if Arg_Index < Ada.Command_Line.Argument_Count then
               Arg_Index := Arg_Index + 1;
               declare
                  Path_Str : constant String := Ada.Command_Line.Argument (Arg_Index);
               begin
                  if Path_Str'Length <= Max_Path_Length then
                     Input_Path := Path_Strings.To_Bounded_String (Path_Str);
                  else
                     Ada.Text_IO.Put_Line ("Error: Input path too long");
                     Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
                     return;
                  end if;
               end;
            else
               Ada.Text_IO.Put_Line ("Error: -i requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         elsif Arg = "-o" then
            if Arg_Index < Ada.Command_Line.Argument_Count then
               Arg_Index := Arg_Index + 1;
               declare
                  Path_Str : constant String := Ada.Command_Line.Argument (Arg_Index);
               begin
                  if Path_Str'Length <= Max_Path_Length then
                     Output_Path := Path_Strings.To_Bounded_String (Path_Str);
                  else
                     Ada.Text_IO.Put_Line ("Error: Output path too long");
                     Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
                     return;
                  end if;
               end;
            else
               Ada.Text_IO.Put_Line ("Error: -o requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         elsif Arg = "-m" then
            if Arg_Index < Ada.Command_Line.Argument_Count then
               Arg_Index := Arg_Index + 1;
               declare
                  Name_Str : constant String := Ada.Command_Line.Argument (Arg_Index);
               begin
                  if Name_Str'Length <= Max_Identifier_Length then
                     Module_Name := Identifier_Strings.To_Bounded_String (Name_Str);
                  else
                     Ada.Text_IO.Put_Line ("Error: Module name too long");
                     Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
                     return;
                  end if;
               end;
            else
               Ada.Text_IO.Put_Line ("Error: -m requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         else
            Ada.Text_IO.Put_Line ("Error: Unknown option: " & Arg);
            Print_Usage;
            Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
            return;
         end if;
         Arg_Index := Arg_Index + 1;
      end;
   end loop;

   --  Validate required arguments
   if Path_Strings.Length (Input_Path) = 0 then
      Ada.Text_IO.Put_Line ("Error: Input path required (-i)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   if Path_Strings.Length (Output_Path) = 0 then
      Ada.Text_IO.Put_Line ("Error: Output path required (-o)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   if Identifier_Strings.Length (Module_Name) = 0 then
      Ada.Text_IO.Put_Line ("Error: Module name required (-m)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  Process the spec file
   Ada.Text_IO.Put_Line ("Converting spec to IR...");
   Ada.Text_IO.Put_Line ("  Input:  " & Path_Strings.To_String (Input_Path));
   Ada.Text_IO.Put_Line ("  Output: " & Path_Strings.To_String (Output_Path));
   Ada.Text_IO.Put_Line ("  Module: " & Identifier_Strings.To_String (Module_Name));

   IR_Converter.Process_Spec_File (Input_Path, Output_Path, Module_Name, Status);

   if Status = Success then
      Ada.Text_IO.Put_Line ("Conversion successful.");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   else
      Ada.Text_IO.Put_Line ("Error: Conversion failed with status " & Status_Code'Image (Status));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;

end IR_Converter_Main;