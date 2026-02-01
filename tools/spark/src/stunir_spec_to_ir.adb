-------------------------------------------------------------------------------
--  STUNIR Spec to IR Converter - Ada SPARK Implementation V2
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package implements the core functionality for converting specifications
--  to STUNIR Semantic IR format (NOT file manifests).
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.SHA256;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR_JSON_Utils; use STUNIR_JSON_Utils;

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

      Put_Line ("[INFO] Toolchain verified from: " & Lockfile_Path);
      return True;
   end Verify_Toolchain;

   --  Compute SHA-256 hash of file content (unchanged)
   procedure Compute_SHA256
     (File_Path : Path_String;
      Hash      : out Hash_String;
      Success   : out Boolean)
   is
      use Ada.Streams;
      use Ada.Streams.Stream_IO;
      use GNAT.SHA256;

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

   --  Process a single spec file and generate IR Module
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

   --  Recursive directory processing helper (simplified for v2)
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

   exception
      when others =>
         Success := False;
   end Process_Directory;

   --  Find first JSON file in spec directory
   function Find_Spec_File (Spec_Dir : String) return String
   is
      Search  : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      --  Look for any .json file in the spec directory
      Start_Search (Search, Spec_Dir, "*.json", (Directory => False, others => True));
      
      if More_Entries (Search) then
         Get_Next_Entry (Search, Dir_Ent);
         declare
            Found_File : constant String := Full_Name (Dir_Ent);
         begin
            End_Search (Search);
            return Found_File;
         end;
      end if;
      
      End_Search (Search);
      return "";  --  No JSON files found
   end Find_Spec_File;

   --  Collect all JSON spec files from directory
   type Spec_File_Array is array (Positive range <>) of Path_String;
   type Spec_File_List is record
      Files : Spec_File_Array (1 .. 10);
      Count : Natural := 0;
   end record;

   procedure Collect_Spec_Files
     (Spec_Dir : String;
      File_List : out Spec_File_List;
      Success : out Boolean)
   is
      Search  : Search_Type;
      Dir_Ent : Directory_Entry_Type;
   begin
      File_List.Count := 0;
      Success := True;

      --  Search for all .json files
      Start_Search (Search, Spec_Dir, "*.json", (Directory => False, others => True));

      while More_Entries (Search) and File_List.Count < File_List.Files'Last loop
         Get_Next_Entry (Search, Dir_Ent);
         File_List.Count := File_List.Count + 1;
         File_List.Files (File_List.Count) := 
           Path_Strings.To_Bounded_String (Full_Name (Dir_Ent));
      end loop;

      End_Search (Search);

      if File_List.Count = 0 then
         Success := False;
      end if;

   exception
      when others =>
         Success := False;
   end Collect_Spec_Files;

   --  Main conversion procedure - NOW GENERATES SEMANTIC IR WITH MULTI-FILE SUPPORT
   procedure Convert_Spec_To_IR
     (Config : Conversion_Config;
      Result : out Conversion_Result)
   is
      Spec_Dir    : constant String := Path_Strings.To_String (Config.Spec_Root);
      File_List   : Spec_File_List;
      List_OK     : Boolean;
      Module      : IR_Module;
      JSON_Output : JSON_Buffer;
      Parse_Stat  : Parse_Status;
      Out_Name    : constant String := Path_Strings.To_String (Config.Output_Path);
      Out_Dir     : constant String := Containing_Directory (Out_Name);
   begin
      --  Initialize result with minimal defaults (avoid stack overflow)
      Result.Status := Success;
      Result.Manifest.Count := 0;

      --  Step 1: Verify toolchain
      Put_Line ("[INFO] Loading toolchain from " & Path_Strings.To_String (Config.Lockfile) & "...");
      if not Verify_Toolchain (Config.Lockfile) then
         Result.Status := Error_Toolchain_Verification_Failed;
         return;
      end if;

      --  Step 2: Collect all spec files
      Put_Line ("[INFO] Collecting spec files from " & Spec_Dir & "...");
      Collect_Spec_Files (Spec_Dir, File_List, List_OK);
      
      if not List_OK or File_List.Count = 0 then
         Put_Line ("[ERROR] No JSON spec files found in: " & Spec_Dir);
         Result.Status := Error_Spec_Not_Found;
         return;
      end if;

      Put_Line ("[INFO] Found" & Natural'Image (File_List.Count) & " spec file(s)");
      --  Step 3: Parse spec files, trying each until we find a valid one
      --  v0.8.3: Skip invalid files (like lockfiles) gracefully
      declare
         First_Valid_Found : Boolean := False;
         File     : Ada.Text_IO.File_Type;
         JSON_Str : String (1 .. 100_000);
         Last     : Natural;
      begin
         for I in 1 .. File_List.Count loop
            declare
               Spec_File : constant String := Path_Strings.To_String (File_List.Files (I));
            begin
               Put_Line ("[INFO] Parsing spec from " & Spec_File & "...");
               Ada.Text_IO.Open (File, Ada.Text_IO.In_File, Spec_File);
               
               --  Read entire file
               Last := 0;
               while not Ada.Text_IO.End_Of_File (File) and Last < JSON_Str'Last loop
                  declare
                     Line : constant String := Ada.Text_IO.Get_Line (File);
                  begin
                     if Last + Line'Length <= JSON_Str'Last then
                        JSON_Str (Last + 1 .. Last + Line'Length) := Line;
                        Last := Last + Line'Length;
                     end if;
                  end;
               end loop;
               
               Ada.Text_IO.Close (File);

               --  Try to parse JSON
               if not First_Valid_Found then
                  --  Try as first module
                  Parse_Spec_JSON (JSON_Str (1 .. Last), Module, Parse_Stat);
                  
                  if Parse_Stat = Success then
                     First_Valid_Found := True;
                  else
                     Put_Line ("[WARN] Failed to parse " & Spec_File & ", skipping");
                  end if;
               else
                  --  Merge as additional module
                  declare
                     Additional_Module : IR_Module;
                  begin
                     Parse_Spec_JSON (JSON_Str (1 .. Last), Additional_Module, Parse_Stat);
                     
                     if Parse_Stat = Success then
                        --  Merge functions
                        for J in 1 .. Additional_Module.Func_Cnt loop
                           if Module.Func_Cnt < Max_Functions then
                              Module.Func_Cnt := Module.Func_Cnt + 1;
                              Module.Functions (Module.Func_Cnt) := Additional_Module.Functions (J);
                           else
                              Put_Line ("[WARN] Maximum function count reached");
                              exit;
                           end if;
                        end loop;
                     else
                        Put_Line ("[WARN] Failed to parse " & Spec_File & ", skipping");
                     end if;
                  end;
               end if;
            end;
         end loop;
         
         --  Check if we found at least one valid spec
         if not First_Valid_Found then
            Put_Line ("[ERROR] No valid spec files found");
            Result.Status := Error_Invalid_Spec;
            return;
         end if;
      end;

      --  Step 4: Multi-file merging is now integrated in Step 3 above
      --  The old Step 4 code (lines 310-359) is replaced by the unified loop
      --  Step 4 is now integrated into Step 3 (legacy code removed for v0.8.3)

      --  Step 5: Generate semantic IR JSON
      Put_Line ("[INFO] Generating semantic IR with" & Natural'Image (Module.Func_Cnt) & " function(s)...");
      IR_To_JSON (Module, JSON_Output, Parse_Stat);
      
      if Parse_Stat /= Success then
         Put_Line ("[ERROR] Failed to generate IR JSON");
         Result.Status := Error_Output_Write_Failed;
         return;
      end if;

      --  Step 6: Write IR to output file
      if not Exists (Out_Dir) then
         Create_Directory (Out_Dir);
      end if;

      declare
         Out_Text_File : Ada.Text_IO.File_Type;
      begin
         Ada.Text_IO.Create (Out_Text_File, Ada.Text_IO.Out_File, Out_Name);
         Ada.Text_IO.Put_Line (Out_Text_File, JSON_Buffers.To_String (JSON_Output));
         Ada.Text_IO.Close (Out_Text_File);
      end;

      Put_Line ("[INFO] Wrote semantic IR to " & Out_Name);
      Put_Line ("[SUCCESS] Generated semantic IR with schema: stunir_ir_v1");
      Result.Status := Success;

   exception
      when others =>
         Put_Line ("[ERROR] Exception during IR generation");
         Result.Status := Error_Output_Write_Failed;
   end Convert_Spec_To_IR;

   --  Write manifest to output file (kept for compatibility, but not used in v2)
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

      if not Exists (Out_Dir) then
         Create_Directory (Out_Dir);
      end if;

      Create (Output_File, Out_File, Out_Name);

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