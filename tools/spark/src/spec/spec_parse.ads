--  Spec Parse Micro-Tool
--  Parses spec JSON into internal representation
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Spec_Parse is

   --  Parsed spec data (extended for full module structure)
   type Spec_Data is record
      Schema_Version : Identifier_String;
      Module_Name    : Identifier_String;
      Imports        : Import_Collection;
      Exports        : Export_Collection;
      Types          : Type_Def_Collection;
      Constants      : Constant_Collection;
      Dependencies   : Dependency_Collection;
      Functions      : Function_Collection;
   end record;

   --  Parse spec JSON file into Spec_Data structure
   procedure Parse_Spec_File
     (Input_Path : in     Path_String;
      Spec       :    out Spec_Data;
      Status     :    out Status_Code);

   --  Parse spec JSON string into Spec_Data structure
   procedure Parse_Spec_String
     (JSON_Content : in     JSON_String;
      Spec         :    out Spec_Data;
      Status       :    out Status_Code);

end Spec_Parse;
