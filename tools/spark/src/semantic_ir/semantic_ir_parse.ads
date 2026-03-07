-------------------------------------------------------------------------------
--  STUNIR Semantic IR Parse Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package provides parsing capabilities for Semantic IR JSON files.
--  It reads Semantic IR JSON and constructs the internal representation.
--
--  Key features:
--  - Parses Semantic IR JSON into Semantic_Module structures
--  - Validates node bindings and type references
--  - Enforces normal form during parsing
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Modules; use Semantic_IR.Modules;
with STUNIR_Types; use STUNIR_Types;

package Semantic_IR.Parse is

   --  Parse Semantic IR JSON file into Semantic_Module structure
   procedure Parse_Semantic_IR_File
     (Input_Path : in     Path_String;
      Module     :    out Semantic_Module;
      Status     :    out Status_Code);

   --  Parse Semantic IR JSON string into Semantic_Module structure
   procedure Parse_Semantic_IR_String
     (JSON_Content : in     JSON_String;
      Module       :    out Semantic_Module;
      Status       :    out Status_Code);

   --  Parse flat IR JSON and convert to Semantic IR
   --  This is the main entry point for the Spec -> IR -> Semantic IR pipeline
   procedure Convert_Flat_IR_To_Semantic
     (Flat_IR_JSON : in     JSON_String;
      Semantic_IR  :    out JSON_String;
      Status       :    out Status_Code);

end Semantic_IR.Parse;