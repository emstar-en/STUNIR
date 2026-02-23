--  spark_extract_main - Minimal SPARK source -> extraction.json
--  Phase 0: Source extraction (SPARK)
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with STUNIR_Types;
with Spark_Extract;

procedure Spark_Extract_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;
   use Spark_Extract;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Module_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
   Language    : Identifier_String := Identifier_Strings.To_Bounded_String ("spark");

   Exit_Failure : constant := 1;
   Version_Str  : constant String := "spark_extract_main@2026-02-23a";

begin
   --  Version banner to stdout, not stderr (stderr reserved for errors only)
   Put_Line (Version_Str);
   if ACL.Argument_Count < 2 then
      Put_Line ("Usage: spark_extract_main -i <source> -o <extraction.json> [-m <module>] [--lang <lang>]");
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
      Output_Path_Str : constant String := Path_Strings.To_String (Output_Path);
      Started_Path    : constant String := Output_Path_Str & ".started.txt";
      Started_Dir     : constant String := Ada.Directories.Containing_Directory (Output_Path_Str);
      Started_File    : File_Type;
   begin
      if Started_Dir'Length > 0 and then not Ada.Directories.Exists (Started_Dir) then
         Ada.Directories.Create_Path (Started_Dir);
      end if;
      Create (Started_File, Out_File, Started_Path);
      Put_Line (Started_File, Version_Str);
      Put_Line (Started_File, "Input:  " & Path_Strings.To_String (Input_Path));
      Put_Line (Started_File, "Output: " & Output_Path_Str);
      Close (Started_File);
   exception
      when others => null;
   end;

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
         declare
            Output_Path_Str : constant String := Path_Strings.To_String (Output_Path);
            Ok_Path : constant String := Output_Path_Str & ".ok.txt";
            Ok_Dir  : constant String := Ada.Directories.Containing_Directory (Output_Path_Str);
            Ok_File : File_Type;
         begin
            if Ok_Dir'Length > 0 and then not Ada.Directories.Exists (Ok_Dir) then
               Ada.Directories.Create_Path (Ok_Dir);
            end if;
            Create (Ok_File, Out_File, Ok_Path);
            Put_Line (Ok_File, Version_Str);
            Close (Ok_File);
         exception
            when others => null;
         end;
         ACL.Set_Exit_Status (ACL.Success);
      else
         Put_Line (Standard_Error, "Error: " & Status_Code_Image (Status));
         if Spark_Extract.Last_Error'Length = 0 then
            Put_Line (Standard_Error, "Detail: (none)");
         else
            Put_Line (Standard_Error, "Detail: " & Spark_Extract.Last_Error);
         end if;
         Put_Line (Standard_Error, "Input:  " & Path_Strings.To_String (Input_Path));
         Put_Line (Standard_Error, "Output: " & Path_Strings.To_String (Output_Path));
         declare
            Diag : File_Type;
            Diag_Path : constant String := Path_Strings.To_String (Output_Path) & ".main_error.txt";
            Diag_Dir  : constant String := Ada.Directories.Containing_Directory (Diag_Path);
         begin
            if Diag_Dir'Length > 0 and then not Ada.Directories.Exists (Diag_Dir) then
               Ada.Directories.Create_Path (Diag_Dir);
            end if;
            Create (Diag, Out_File, Diag_Path);
            Put_Line (Diag, "Error: " & Status_Code_Image (Status));
            Put_Line (Diag, "Detail: " & Spark_Extract.Last_Error);
            Put_Line (Diag, "Input:  " & Path_Strings.To_String (Input_Path));
            Put_Line (Diag, "Output: " & Path_Strings.To_String (Output_Path));
            Close (Diag);
         exception
            when others => null;
         end;
         declare
            Output_Path_Str : constant String := Path_Strings.To_String (Output_Path);
            Error_Path : constant String := Output_Path_Str & ".error.txt";
            Error_Dir  : constant String := Ada.Directories.Containing_Directory (Error_Path);
            Err : File_Type;
         begin
            if Error_Dir'Length > 0 and then not Ada.Directories.Exists (Error_Dir) then
               Ada.Directories.Create_Path (Error_Dir);
            end if;
            Create (Err, Out_File, Error_Path);
            Put_Line (Err, "Error: " & Status_Code_Image (Status));
            Put_Line (Err, "Detail: " & Spark_Extract.Last_Error);
            Put_Line (Err, "Input:  " & Path_Strings.To_String (Input_Path));
            Put_Line (Err, "Output: " & Output_Path_Str);
            Close (Err);
         exception
            when others => null;
         end;
         ACL.Set_Exit_Status (ACL.Failure);
      end if;
   end;
exception
   when others =>
      Put_Line ("Error: spark_extract_main failed");
      ACL.Set_Exit_Status (Exit_Failure);
end Spark_Extract_Main;
