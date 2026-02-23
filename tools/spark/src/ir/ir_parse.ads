--  IR Parse Micro-Tool
--  Parses IR JSON into internal representation
--  Phase: 2 (IR)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package IR_Parse is

   --  Parse IR JSON file into IR_Data structure
   procedure Parse_IR_File
     (Input_Path : in     Path_String;
      IR         :    out IR_Data;
      Status     :    out Status_Code);

   --  Parse IR JSON string into IR_Data structure
   procedure Parse_IR_String
     (JSON_Content : in     JSON_String;
      IR           :    out IR_Data;
      Status       :    out Status_Code);

end IR_Parse;
