-------------------------------------------------------------------------------
--  STUNIR Code Index - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Indexes source code files and produces a SHA-256 keyed JSON manifest.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Code_Index is

   Max_Path_Length : constant := 512;
   Max_Hash_Length : constant := 64;
   Max_Files : constant := 5000;

   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   type Language_Kind is
     (LANG_C, LANG_CPP, LANG_ADA, LANG_RUST, LANG_PYTHON, LANG_JAVA, LANG_UNKNOWN);

   type File_Entry is record
      Relative_Path : Path_String;
      Absolute_Path : Path_String;
      SHA256        : Hash_String;
      Size_Bytes    : Natural := 0;
      Language      : Language_Kind := LANG_UNKNOWN;
   end record;

   type File_Entry_Array is array (Positive range <>) of File_Entry;

   type Code_Index is record
      Root_Path  : Path_String;
      Index_Hash : Hash_String;
      File_Count : Natural := 0;
      Files      : File_Entry_Array (1 .. Max_Files);
   end record;

   type Index_Status is
     (Success,
      Error_Input_Not_Found,
      Error_Output_Failed,
      Error_Index_Overflow,
      Error_Hash_Failed);

   type Index_Result is record
      Status : Index_Status := Success;
      Index  : Code_Index;
   end record;

   type Index_Config is record
      Input_Dir  : Path_String;
      Output_Path : Path_String;
   end record;

   procedure Run_Code_Index;

   procedure Index_Directory
     (Config : Index_Config;
      Result : in out Index_Result);

   function Detect_Language (File_Path : String) return Language_Kind;

   procedure Compute_File_Hash
     (File_Path : String;
      Hash      : out Hash_String;
      Success   : out Boolean);

   procedure Compute_Index_Hash
     (Index  : in out Code_Index;
      Success : out Boolean);

   procedure Write_Index_JSON
     (Index   : Code_Index;
      Output_Path : String;
      Success : out Boolean);

end STUNIR_Code_Index;
