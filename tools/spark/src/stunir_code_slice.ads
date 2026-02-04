-------------------------------------------------------------------------------
--  STUNIR Code Slice - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Slices a source file into detected code regions for AI extraction.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Code_Slice is

   Max_Path_Length : constant := 512;
   Max_Hash_Length : constant := 64;
   Max_Lines : constant := 20000;
   Max_Regions : constant := 2000;

   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   package Lang_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => 16);
   subtype Lang_String is Lang_Strings.Bounded_String;

   type Region_Type is (REGION_FUNCTION, REGION_TYPE_DEF, REGION_CONSTANT, REGION_UNKNOWN);

   type Code_Region is record
      Start_Line  : Natural := 0;
      End_Line    : Natural := 0;
      Region_Kind : Region_Type := REGION_UNKNOWN;
      Content_Hash : Hash_String;
   end record;

   type Region_Array is array (Positive range <>) of Code_Region;

   type Sliced_File is record
      File_Path   : Path_String;
      File_Hash   : Hash_String;
      Language    : Lang_String;
      Region_Count : Natural := 0;
      Regions     : Region_Array (1 .. Max_Regions);
   end record;

   type Slice_Status is (Success, Error_Input_Not_Found, Error_Output_Failed, Error_Parse_Failed);

   type Slice_Result is record
      Status : Slice_Status := Success;
      Slice  : Sliced_File;
   end record;

   type Slice_Config is record
      Input_File : Path_String;
      Output_Path : Path_String;
      Index_Path  : Path_String;
      Has_Index   : Boolean := False;
   end record;

   procedure Run_Code_Slice;

   procedure Slice_File
     (Config : Slice_Config;
      Result : in out Slice_Result);

end STUNIR_Code_Slice;
