--  Code Emitter Main Program
--  Command-line entry point for ir.json to target code conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);  --  Command-line parsing not in SPARK

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
use STUNIR_Types;
with Code_Emitter;

procedure Code_Emitter_Main is

   procedure Print_Usage is
   begin
      Ada.Text_IO.Put_Line ("Usage: code_emitter -i <input> -o <output> -t <target>");
      Ada.Text_IO.Put_Line ("  -i <input>    Input ir.json file path");
      Ada.Text_IO.Put_Line ("  -o <output>   Output directory for generated code");
      Ada.Text_IO.Put_Line ("  -t <target>   Target language (or 'all' for all targets)");
      Ada.Text_IO.Put_Line ("  -h            Show this help message");
      Ada.Text_IO.Put_Line ("");
      Ada.Text_IO.Put_Line ("Supported targets:");
      Ada.Text_IO.Put_Line ("  cpp, c, python, rust, go, java, javascript, typescript, ada, spark, all");
   end Print_Usage;

   function Parse_Target (Target_Str : String) return Target_Language is
   begin
      if Target_Str = "cpp" or Target_Str = "c++" then
         return Lang_CPP;
      elsif Target_Str = "c" then
         return Lang_C;
      elsif Target_Str = "python" or Target_Str = "py" then
         return Lang_Python;
      elsif Target_Str = "rust" or Target_Str = "rs" then
         return Lang_Rust;
      elsif Target_Str = "go" or Target_Str = "golang" then
         return Lang_Go;
      elsif Target_Str = "java" then
         return Lang_Java;
      elsif Target_Str = "javascript" or Target_Str = "js" then
         return Lang_JavaScript;
      elsif Target_Str = "typescript" or Target_Str = "ts" then
         return Lang_TypeScript;
      elsif Target_Str = "ada" then
         return Lang_Ada;
      elsif Target_Str = "spark" then
         return Lang_SPARK;
      else
         return Lang_CPP;  --  Default
      end if;
   end Parse_Target;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Dir  : Path_String := Path_Strings.Null_Bounded_String;
   Target_Str  : Identifier_String := Identifier_Strings.Null_Bounded_String;
   Target      : Target_Language := Lang_CPP;
   All_Targets : Boolean := False;
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
                     Output_Dir := Path_Strings.To_Bounded_String (Path_Str);
                  else
                     Ada.Text_IO.Put_Line ("Error: Output directory too long");
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
         elsif Arg = "-t" then
            if Arg_Index < Ada.Command_Line.Argument_Count then
               Arg_Index := Arg_Index + 1;
               declare
                  T_Str : constant String := Ada.Command_Line.Argument (Arg_Index);
               begin
                  if T_Str = "all" then
                     All_Targets := True;
                  else
                     if T_Str'Length <= Max_Identifier_Length then
                        Target_Str := Identifier_Strings.To_Bounded_String (T_Str);
                        Target := Parse_Target (T_Str);
                     else
                        Ada.Text_IO.Put_Line ("Error: Target name too long");
                        Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
                        return;
                     end if;
                  end if;
               end;
            else
               Ada.Text_IO.Put_Line ("Error: -t requires an argument");
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

   if Path_Strings.Length (Output_Dir) = 0 then
      Ada.Text_IO.Put_Line ("Error: Output directory required (-o)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   if not All_Targets and Identifier_Strings.Length (Target_Str) = 0 then
      Ada.Text_IO.Put_Line ("Error: Target language required (-t) or use 'all'");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  Process the IR file
   Ada.Text_IO.Put_Line ("Generating target code...");
   Ada.Text_IO.Put_Line ("  Input:  " & Path_Strings.To_String (Input_Path));
   Ada.Text_IO.Put_Line ("  Output: " & Path_Strings.To_String (Output_Dir));

   if All_Targets then
      Ada.Text_IO.Put_Line ("  Target: all languages");
      Code_Emitter.Process_IR_File_All_Targets (Input_Path, Output_Dir, Status);
   else
      Ada.Text_IO.Put_Line ("  Target: " & Identifier_Strings.To_String (Target_Str));
      Code_Emitter.Process_IR_File (Input_Path, Output_Dir, Target, Status);
   end if;

   if Status = Success then
      Ada.Text_IO.Put_Line ("Code generation successful.");
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   else
      Ada.Text_IO.Put_Line ("Error: Code generation failed with status " & Status_Code'Image (Status));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;

end Code_Emitter_Main;