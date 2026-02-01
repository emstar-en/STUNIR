-------------------------------------------------------------------------------
--  STUNIR JSON Utilities - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Lightweight JSON parsing and generation for DO-178C Level A compliance
--  Focuses on the specific JSON structures needed for STUNIR semantic IR
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;

package STUNIR_JSON_Utils is

   Max_JSON_Size : constant := 1_048_576;  -- 1 MB max JSON
   
   package JSON_Buffers is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_JSON_Size);
   subtype JSON_Buffer is JSON_Buffers.Bounded_String;

   --  Parse status
   type Parse_Status is (Success, Error_Invalid_JSON, Error_Missing_Field, 
                         Error_Too_Large, Error_Invalid_Type);

   --  Parse simple JSON spec into IR Module
   procedure Parse_Spec_JSON
     (JSON_Text : String;
      Module    : out IR_Module;
      Status    : out Parse_Status)
   with
     Pre => JSON_Text'Length > 0 and JSON_Text'Length <= Max_JSON_Size,
     Post => (if Status = Success then Module.Func_Cnt > 0 or Module.Type_Cnt >= 0);

   --  Serialize IR Module to JSON
   procedure IR_To_JSON
     (Module : IR_Module;
      Output : out JSON_Buffer;
      Status : out Parse_Status)
   with
     Pre => Is_Valid_Module (Module),
     Post => (if Status = Success then JSON_Buffers.Length (Output) > 0);

   --  Extract string value from JSON field (helper)
   function Extract_String_Value
     (JSON_Text : String;
      Field_Name : String) return String
   with
     Pre => JSON_Text'Length > 0 and Field_Name'Length > 0,
     Post => Extract_String_Value'Result'Length <= 1024;

   --  Extract integer value from JSON field (helper)
   --  Returns 0 if field not found or invalid
   function Extract_Integer_Value
     (JSON_Text : String;
      Field_Name : String) return Natural
   with
     Pre => JSON_Text'Length > 0 and Field_Name'Length > 0;

   --  Find array in JSON and return position of opening bracket
   function Find_Array (JSON_Text : String; Field : String) return Natural
   with
     Pre => JSON_Text'Length > 0 and Field'Length > 0;

   --  Extract next object from array at position
   procedure Get_Next_Object
     (JSON_Text : String;
      Start_Pos : Natural;
      Obj_Start : out Natural;
      Obj_End   : out Natural)
   with
     Pre => JSON_Text'Length > 0 and Start_Pos <= JSON_Text'Last;

   --  Compute SHA-256 hash of JSON (deterministic)
   function Compute_JSON_Hash (JSON_Text : String) return String
   with
     Pre => JSON_Text'Length > 0,
     Post => Compute_JSON_Hash'Result'Length = 64;

end STUNIR_JSON_Utils;
