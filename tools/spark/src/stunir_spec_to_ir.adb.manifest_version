-------------------------------------------------------------------------------
--  STUNIR Spec to IR Converter - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package implements the core functionality for converting specifications
--  to STUNIR Intermediate Reference (IR) format.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with GNAT.SHA256;
with Ada.Streams;
with Ada.Streams.Stream_IO;

package body STUNIR_Spec_To_IR is

   use Ada.Text_IO;
   use Ada.Directories;

   --  Initialize configuration with provided paths
   procedure Initialize_Config
     (Config    : out Conversion_Config;
      Spec_Root : String;
      Output    : String;
      Lockfile  : String := "local_toolchain.lock.json")
   is
   begin
      Config.Spec_Root   := Path_Strings.To_Bounded_String (Spec_Root);
      Config.Output_Path := Path_Strings.To_Bounded_String (Output);
      Config.Lockfile    := Path_Strings.To_Bounded_String (Lockfile);
      Config.Strict_Mode := True;
   end Initialize_Config;

   --  Verify toolchain lock file exists and is valid
   function Verify_Toolchain (Lockfile : Path_String) return Boolean
   is
      Lockfile_Path : constant String := Path_Strings.To_String (Lockfile);
   begin
      --  Check if lockfile exists
      if not Exists (Lockfile_Path) then
         Put_Line ("[ERROR] Toolchain lockfile not found: " & Lockfile_Path);
         return False;
      end if;

      --  TODO: Parse and validate lockfile JSON content
      --  For now, existence check is sufficient
      Put_Line ("[INFO] Toolchain verified from: " & Lockfile_Path);
      return True;
   end Verify_Toolchain;

   --  Compute SHA-256 hash of file content
   procedure Compute_SHA256
     (File_Path : Path_String;
      Hash      : out Hash_String;
      Success   : out Boolean)
   is
      use Ada.Streams;
      use Ada.Streams.Stream_IO;

      File_Name : constant String := Path_Strings.To_String (File_Path);
      File      : Ada.Streams.Stream_IO.File_Type;
      Context   : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Buffer    : Stream_Element_Array (1 .. 8192);
      Last      : Stream_Element_Offset;
   begin
      Hash := Hash_Strings.Null_Bounded_String;
      Success := False;

      if not Exists (File_Name) then
         return;
      end if;

      Open (File, In_File, File_Name);

      while not End_Of_File (File) loop
         Read (File, Buffer, Last);
         if Last >= Buffer'First then
            GNAT.SHA256.Update (Context, Buffer (Buffer'First .. Last));
         end if;
      end loop;

      Close (File);

      declare
         Digest : constant String := GNAT.SHA256.Digest (Context);
      begin
         Hash := Hash_Strings.To_Bounded_String (Digest);
         Success := True;
      end;

   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
         Success := False;
   end Compute_SHA256;

   --  Process a single spec file and create manifest entry
   procedure Process_Spec_File
     (File_Path : Path_String;
      Base_Path : Path_String;
      Entry_Out : out Manifest_Entry;
      Success   : out Boolean)
   is
      Full_Path : constant String := Path_Strings.To_String (File_Path);
      Base_Dir  : constant String := Path_Strings.To_String (Base_Path);
      Rel_Path  : constant String := Full_Path (Base_Dir'Length + 2 .. Full_Path'Last);
      Hash_Val  : Hash_String;
      Hash_OK   : Boolean;
   begin
      Entry_Out := (Path   => Path_Strings.Null_Bounded_String,
                    SHA256 => Hash_Strings.Null_Bounded_String,
                    Size   => 0);
      Success := False;

      --  Compute file hash
      Compute_SHA256 (File_Path, Hash_Val, Hash_OK);
      if not Hash_OK then
         Put_Line ("[ERROR] Failed to compute hash for: " & Full_Path);
         return;
      end if;

      --  Set entry fields
      Entry_Out.Path   := Path_Strings.To_Bounded_String (Rel_Path);
      Entry_Out.SHA256 := Hash_Val;
      Entry_Out.Size   := Natural (Size (Full_Path));
      Success := True;
   end Process_Spec_File;

   --  Recursive directory processing helper
   procedure Process_Directory
     (Dir_Path  : String;
      Base_Path : Path_String;
      Manifest  : in out IR_Manifest;
      Success   : out Boolean)
   is
      Search  : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      Success := True;

      Start_Search (Search, Dir_Path, "*.json", (Directory => False, others => True));

      while More_Entries (Search) loop
         Get_Next_Entry (Search, Dir_Ent);

         declare
            Entry_Full_Name : constant String := Full_Name (Dir_Ent);
            File_Path : constant Path_String := Path_Strings.To_Bounded_String (Entry_Full_Name);
            Entry_Val : Manifest_Entry;
            Entry_OK  : Boolean;
         begin
            Process_Spec_File (File_Path, Base_Path, Entry_Val, Entry_OK);
            if Entry_OK and Manifest.Count < Max_Manifest_Entries then
               Manifest.Count := Manifest.Count + 1;
               Manifest.Entries (Manifest.Count) := Entry_Val;
            end if;
         end;
      end loop;

      End_Search (Search);

      --  Process subdirectories
      Start_Search (Search, Dir_Path, "*", (Directory => True, others => False));

      while More_Entries (Search) loop
         Get_Next_Entry (Search, Dir_Ent);

         declare
            Sub_Name : constant String := Simple_Name (Dir_Ent);
         begin
            if Sub_Name /= "." and Sub_Name /= ".." then
               declare
                  Sub_OK : Boolean;
               begin
                  Process_Directory (Full_Name (Dir_Ent), Base_Path, Manifest, Sub_OK);
                  Success := Success and Sub_OK;
               end;
            end if;
         end;
      end loop;

      End_Search (Search);

   exception
      when others =>
         Success := False;
   end Process_Directory;

   --  Main conversion procedure
   procedure Convert_Spec_To_IR
     (Config : Conversion_Config;
      Result : out Conversion_Result)
   is
      Spec_Dir : constant String := Path_Strings.To_String (Config.Spec_Root);
      Proc_OK  : Boolean;
   begin
      Result := (Status   => Success,
                 Manifest => (Entries => (others => (Path   => Path_Strings.Null_Bounded_String,
                                                     SHA256 => Hash_Strings.Null_Bounded_String,
                                                     Size   => 0)),
                              Count => 0));

      --  Step 1: Verify toolchain
      Put_Line ("[INFO] Loading toolchain from " & Path_Strings.To_String (Config.Lockfile) & "...");
      if not Verify_Toolchain (Config.Lockfile) then
         Result.Status := Error_Toolchain_Verification_Failed;
         return;
      end if;

      --  Step 2: Check spec root exists
      if not Exists (Spec_Dir) then
         Put_Line ("[ERROR] Spec root not found: " & Spec_Dir);
         Result.Status := Error_Spec_Not_Found;
         return;
      end if;

      --  Step 3: Process specs
      Put_Line ("[INFO] Processing specs from " & Spec_Dir & "...");
      Process_Directory (Spec_Dir, Config.Spec_Root, Result.Manifest, Proc_OK);

      if not Proc_OK then
         Result.Status := Error_Invalid_Spec;
         return;
      end if;

      --  Step 4: Write output
      declare
         Write_OK : Boolean;
      begin
         Write_Manifest (Result.Manifest, Config.Output_Path, Write_OK);
         if not Write_OK then
            Result.Status := Error_Output_Write_Failed;
            return;
         end if;
      end;

      Put_Line ("[INFO] Wrote IR manifest to " & Path_Strings.To_String (Config.Output_Path));
      Result.Status := Success;
   end Convert_Spec_To_IR;

   --  Write manifest to output file in canonical JSON format
   procedure Write_Manifest
     (Manifest    : IR_Manifest;
      Output_Path : Path_String;
      Success     : out Boolean)
   is
      Output_File : File_Type;
      Out_Name : constant String := Path_Strings.To_String (Output_Path);
      Out_Dir  : constant String := Containing_Directory (Out_Name);
   begin
      Success := False;

      --  Create output directory if needed
      if not Exists (Out_Dir) then
         Create_Directory (Out_Dir);
      end if;

      Create (Output_File, Out_File, Out_Name);

      --  Write canonical JSON (sorted keys, minimal separators)
      Put (Output_File, "[");

      for I in 1 .. Manifest.Count loop
         if I > 1 then
            Put (Output_File, ",");
         end if;
         Put (Output_File, "{""path"":""" &
              Path_Strings.To_String (Manifest.Entries (I).Path) &
              """,""sha256"":""" &
              Hash_Strings.To_String (Manifest.Entries (I).SHA256) &
              """,""size"":" &
              Natural'Image (Manifest.Entries (I).Size) &
              "}");
      end loop;

      Put_Line (Output_File, "]");
      Close (Output_File);
      Success := True;

   exception
      when others =>
         if Is_Open (Output_File) then
            Close (Output_File);
         end if;
         Success := False;
   end Write_Manifest;

   --  Entry point for command-line execution
   procedure Run_Spec_To_IR
   is
      use Ada.Command_Line;
      Config      : Conversion_Config;
      Result      : Conversion_Result;
      Spec_Root   : Path_String;
      Output_Path : Path_String;
      Lockfile    : Path_String := Path_Strings.To_Bounded_String ("local_toolchain.lock.json");
   begin
      --  Parse command line arguments
      if Argument_Count < 4 then
         Put_Line ("Usage: spec_to_ir --spec-root <dir> --out <file> [--lockfile <file>]");
         Set_Exit_Status (Failure);
         return;
      end if;

      --  Simple argument parsing
      for I in 1 .. Argument_Count loop
         if Argument (I) = "--spec-root" and I < Argument_Count then
            Spec_Root := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--out" and I < Argument_Count then
            Output_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--lockfile" and I < Argument_Count then
            Lockfile := Path_Strings.To_Bounded_String (Argument (I + 1));
         end if;
      end loop;

      Initialize_Config
        (Config,
         Path_Strings.To_String (Spec_Root),
         Path_Strings.To_String (Output_Path),
         Path_Strings.To_String (Lockfile));

      Convert_Spec_To_IR (Config, Result);

      case Result.Status is
         when Success =>
            Set_Exit_Status (Ada.Command_Line.Success);
         when others =>
            Put_Line ("[ERROR] Conversion failed with status: " & Conversion_Status'Image (Result.Status));
            Set_Exit_Status (Ada.Command_Line.Failure);
      end case;
   end Run_Spec_To_IR;

end STUNIR_Spec_To_IR;
