-------------------------------------------------------------------------------
--  STUNIR Spec to IR Converter - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package provides the core functionality for converting specifications
--  to STUNIR Intermediate Reference (IR) format.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Spec_To_IR is

   --  Maximum path length
   Max_Path_Length : constant := 512;

   --  Maximum hash length (SHA-256 hex = 64 chars)
   Max_Hash_Length : constant := 64;

   --  Maximum manifest entries
   Max_Manifest_Entries : constant := 500;

   --  Bounded string types
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   --  Manifest entry record
   type Manifest_Entry is record
      Path    : Path_String;
      SHA256  : Hash_String;
      Size    : Natural := 0;
   end record;

   --  Manifest array type
   type Manifest_Array is array (Positive range <>) of Manifest_Entry;

   --  Manifest record with count
   type IR_Manifest is record
      Entries : Manifest_Array (1 .. Max_Manifest_Entries);
      Count   : Natural := 0;
   end record;

   --  Conversion result type
   type Conversion_Status is
     (Success,
      Error_Spec_Not_Found,
      Error_Invalid_Spec,
      Error_Toolchain_Verification_Failed,
      Error_Output_Write_Failed,
      Error_Hash_Computation_Failed);

   type Conversion_Result is record
      Status   : Conversion_Status := Success;
      Manifest : IR_Manifest;
   end record;

   --  Configuration record
   type Conversion_Config is record
      Spec_Root    : Path_String;
      Output_Path  : Path_String;
      Lockfile     : Path_String;
      Strict_Mode  : Boolean := True;
   end record;

   --  Initialize conversion configuration
   procedure Initialize_Config
     (Config    : out Conversion_Config;
      Spec_Root : String;
      Output    : String;
      Lockfile  : String := "local_toolchain.lock.json")
   with
     Pre => Spec_Root'Length <= Max_Path_Length and
            Output'Length <= Max_Path_Length and
            Lockfile'Length <= Max_Path_Length,
     Post => Path_Strings.Length (Config.Spec_Root) = Spec_Root'Length;

   --  Verify toolchain lock file
   function Verify_Toolchain (Lockfile : Path_String) return Boolean;

   --  Compute SHA-256 hash of file content
   procedure Compute_SHA256
     (File_Path : Path_String;
      Hash      : out Hash_String;
      Success   : out Boolean)
   with
     Post => (if Success then Hash_Strings.Length (Hash) = Max_Hash_Length
              else Hash_Strings.Length (Hash) = 0);

   --  Process a single spec file
   procedure Process_Spec_File
     (File_Path : Path_String;
      Base_Path : Path_String;
      Entry_Out : out Manifest_Entry;
      Success   : out Boolean);

   --  Main conversion procedure
   procedure Convert_Spec_To_IR
     (Config : Conversion_Config;
      Result : out Conversion_Result)
   with
     Post => (if Result.Status = Success
              then Result.Manifest.Count >= 0);

   --  Write manifest to output file in canonical JSON format
   procedure Write_Manifest
     (Manifest    : IR_Manifest;
      Output_Path : Path_String;
      Success     : out Boolean);

   --  Entry point for command-line execution
   procedure Run_Spec_To_IR;

end STUNIR_Spec_To_IR;
