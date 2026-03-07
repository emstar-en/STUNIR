--  STUNIR Multi-Format Extraction Parser
--  Parses and normalizes different extraction formats into unified schema
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;

package Extraction_Parser is

   pragma Pure;

   --  ========================================================================
   --  Extraction Format Types
   --  ========================================================================

   type Extraction_Format is (
      Format_Direct_Functions,  --  Functions array with source_file per function
      Format_Files_Array,       --  Files array with nested functions
      Format_Placeholder,       --  Minimal placeholder format
      Format_Legacy_V1,         --  Legacy v1 format
      Format_Unknown            --  Unrecognized format
   );

   --  ========================================================================
   --  Unified Extraction Data
   --  ========================================================================

   type Unified_Function is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
      Source_File : Path_String;
   end record;

   type Function_Array is array (Function_Index range <>) of Unified_Function;

   type Unified_Extraction (Function_Count : Function_Index) is record
      Schema_Version : Identifier_String;
      Format_Detected : Extraction_Format;
      Functions      : Function_Array (1 .. Function_Count);
      Status         : Status_Code;
   end record;

   --  ========================================================================
   --  Format Detection
   --  ========================================================================

   function Detect_Extraction_Format
     (JSON_Content : JSON_String) return Extraction_Format
   with
      Pre => JSON_Strings.Length (JSON_Content) > 0;

   --  ========================================================================
   --  Format-Specific Parsers
   --  ========================================================================

   procedure Parse_Direct_Functions_Format
     (Parser     : in out Parser_State;
      Extraction :    out Unified_Extraction;
      Status     :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length,
      Post => Extraction.Function_Count <= Max_Functions;

   procedure Parse_Files_Array_Format
     (Parser     : in out Parser_State;
      Extraction :    out Unified_Extraction;
      Status     :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length,
      Post => Extraction.Function_Count <= Max_Functions;

   procedure Parse_Placeholder_Format
     (Parser     : in out Parser_State;
      Extraction :    out Unified_Extraction;
      Status     :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length;

   procedure Parse_Legacy_V1_Format
     (Parser     : in out Parser_State;
      Extraction :    out Unified_Extraction;
      Status     :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length,
      Post => Extraction.Function_Count <= Max_Functions;

   --  ========================================================================
   --  Unified Parser Entry Point
   --  ========================================================================

   procedure Parse_Extraction
     (JSON_Content : in     JSON_String;
      Extraction   :    out Unified_Extraction;
      Status       :    out Status_Code)
   with
      Pre  => JSON_Strings.Length (JSON_Content) > 0,
      Post => Extraction.Function_Count <= Max_Functions;

   --  ========================================================================
   --  Validation
   --  ========================================================================

   procedure Validate_Extraction
     (Extraction : in     Unified_Extraction;
      Status     :    out Status_Code);

end Extraction_Parser;