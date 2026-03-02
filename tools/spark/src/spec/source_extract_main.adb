--  source_extract_main - Unified source -> extraction.json entry point
--  Phase 0: Source extraction
--  Routes to appropriate extractor based on --lang flag
--  SPARK_Mode: Off (file I/O + routing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Characters.Handling;
with STUNIR_Types;
with Source_Extract;
with Spark_Extract;

procedure Source_Extract_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;
   use STUNIR_Types;
   use Spark_Extract;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Module_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
   Language    : Identifier_String := Identifier_Strings.To_Bounded_String ("c");

   Exit_Failure : constant := 1;
   Version_Str  : constant String := "source_extract_main@2026-03-01a";

   --  Check if language is Ada/SPARK family
   function Is_Ada_Family (Lang : String) return Boolean is
      L : constant String := Ada.Characters.Handling.To_Lower (Lang);
   begin
      return L = "ada" or else L = "spark" or else L = "adb" or else L = "ads";
   end Is_Ada_Family;

   --  Check if language is C family
   function Is_C_Family (Lang : String) return Boolean is
      L : constant String := Ada.Characters.Handling.To_Lower (Lang);
   begin
      return L = "c" or else L = "cpp" or else L = "c++" or else L = "cc" or else L = "h";
   end Is_C_Family;

begin
   --  Version banner to stdout
   Put_Line (Version_Str);
   if ACL.Argument_Count < 2 then
      Put_Line ("Usage: source_extract_main -i <source> -o <extraction.json> [-m <module>] [--lang <lang>]");
      Put_Line ("");
      Put_Line ("Supported languages:");
      Put_Line ("  c, cpp, c++  - C/C++ source files");
      Put_Line ("  ada, spark   - Ada/SPARK source files");
      Put_Line ("  rust         - Rust source files (use rust_extract_main)");
      Put_Line ("  python       - Python source files (use python_extract_main)");
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
      Lang_Str : constant String := Identifier_Strings.To_String (Language);
   begin
      --  Route to appropriate extractor based on language
      if Is_Ada_Family (Lang_Str) then
         --  For Ada/SPARK, use the spark_extract package
         Spark_Extract.Extract_File
           (Input_Path  => Input_Path,
            Output_Path => Output_Path,
            Module_Name => Module_Name,
            Language    => Language,
            Status      => Status);
      elsif Is_C_Family (Lang_Str) then
         --  C/C++ extraction
         Source_Extract.Extract_File
           (Input_Path  => Input_Path,
            Output_Path => Output_Path,
            Module_Name => Module_Name,
            Language    => Language,
            Status      => Status);
      else
         --  Default: try source_extract
         Put_Line (Standard_Error, "INFO: Unknown language '" & Lang_Str & "' - using C-style extraction");
         Source_Extract.Extract_File
           (Input_Path  => Input_Path,
            Output_Path => Output_Path,
            Module_Name => Module_Name,
            Language    => Language,
            Status      => Status);
      end if;

      if Status = Success then
         ACL.Set_Exit_Status (ACL.Success);
      else
         Put_Line (Standard_Error, "Error: " & Status_Code_Image (Status));
         ACL.Set_Exit_Status (ACL.Failure);
      end if;
   end;
exception
   when others =>
      Put_Line (Standard_Error, "Error: source_extract_main failed");
      ACL.Set_Exit_Status (Exit_Failure);
end Source_Extract_Main;
