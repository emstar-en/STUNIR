--  Extract Parse Micro-Tool
--  Parses extraction JSON into internal representation
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Extract_Parse is

   --  Extraction data types
   type Extract_Function is record
      Name       : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
   end record;

   type Extract_Function_Array is array (Function_Index range <>) of Extract_Function;

   type Extract_Function_Collection is record
      Functions : Extract_Function_Array (1 .. Max_Functions);
      Count     : Function_Index;
   end record;

   type Extract_Data is record
      Module_Name : Identifier_String;
      Functions   : Extract_Function_Collection;
   end record;

   --  Parse extraction JSON file into Extract_Data structure
   procedure Parse_Extract_File
     (Input_Path : in     Path_String;
      Extract    :    out Extract_Data;
      Status     :    out Status_Code);

   --  Parse extraction JSON string into Extract_Data structure
   procedure Parse_Extract_String
     (JSON_Content : in     JSON_String;
      Extract      :    out Extract_Data;
      Status       :    out Status_Code);

end Extract_Parse;
