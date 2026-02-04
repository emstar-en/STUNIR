-------------------------------------------------------------------------------
--  STUNIR Receipt Link - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Links assembled specs to source indexes and AI-provided links.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Receipt_Link is

   Max_Path_Length : constant := 512;
   Max_Name_Length : constant := 128;
   Max_Hash_Length : constant := 64;
   Max_Links       : constant := 2048;

   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Path_Length);
   subtype Path_String is Path_Strings.Bounded_String;

   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Name_Length);
   subtype Name_String is Name_Strings.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   type Link_Entry is record
      Spec_Element : Name_String;
      Source_File  : Path_String;
      Source_Hash  : Hash_String;
      Source_Lines : Name_String;
      Confidence   : Natural := 0;
   end record;

   type Link_Array is array (Positive range <>) of Link_Entry;

   type Receipt_Data is record
      Spec_Path   : Path_String;
      Index_Path  : Path_String;
      Links_Path  : Path_String;
      Output_Path : Path_String;
      Link_Count  : Natural := 0;
      Links       : Link_Array (1 .. Max_Links);
      Receipt_Hash : Hash_String;
   end record;

   type Receipt_Status is
     (Success,
      Error_Input_Not_Found,
      Error_Parse_Failed,
      Error_Output_Failed,
      Error_Link_Invalid);

   type Receipt_Result is record
      Status : Receipt_Status := Success;
      Receipt : Receipt_Data;
   end record;

   procedure Run_Receipt_Link;

   procedure Link_Receipt
     (Spec_Path  : String;
      Index_Path : String;
      Links_Path : String;
      Output_Path : String;
      Result : in out Receipt_Result);

end STUNIR_Receipt_Link;
