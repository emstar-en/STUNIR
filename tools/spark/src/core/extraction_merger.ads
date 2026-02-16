--  STUNIR Extraction Merger
--  Combines function signatures from multiple extraction sources/tools
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package implements a universal extraction combiner that merges
--  function signature information from multiple extraction methods/tools.
--  Each extraction source provides partial information, and the merger
--  combines them into a unified, high-confidence extraction.

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

with Extraction_Parser;
use Extraction_Parser;

package Extraction_Merger is

   pragma Pure;

   --  ========================================================================
   --  Extraction Source Types
   --  ========================================================================

   type Extraction_Method is (
      Method_Clang_AST,       --  Clang AST parser
      Method_CTags,           --  ctags-based extraction
      Method_Cppcheck,        --  Cppcheck analysis
      Method_GCC_XML,         --  GCC-XML output
      Method_CastXML,         --  CastXML output
      Method_Custom_Parser,   --  Custom extraction tool
      Method_LibClang,        --  libClang Python bindings
      Method_Tree_Sitter      --  Tree-sitter parser
   );

   type Confidence_Level is range 0 .. 100;

   --  ========================================================================
   --  Merged Function Information
   --  ========================================================================

   type Type_Source is record
      Type_Name  : Type_Name_String;
      Source     : Extraction_Method;
      Confidence : Confidence_Level;
   end record;

   type Type_Source_Array is
      array (Natural range <>) of Type_Source;

   type Parameter_Merge_Info is record
      Name           : Identifier_String;
      Final_Type     : Type_Name_String;
      Type_Sources   : Natural;  --  Number of sources agreeing
      Confidence     : Confidence_Level;
   end record;

   type Merged_Parameter_Array is
      array (Parameter_Index range <>) of Parameter_Merge_Info;

   type Merged_Function is record
      Name              : Identifier_String;
      Return_Type       : Type_Name_String;
      Parameters        : Parameter_List;
      Source_File       : Path_String;
      Extraction_Count  : Natural;  --  How many sources found this function
      Type_Confidence   : Confidence_Level;
      Signature_Hash    : Identifier_String;  --  For detecting changes
   end record;

   --  ========================================================================
   --  Multi-Source Extraction Collection
   --  ========================================================================

   Max_Sources : constant := 8;

   type Source_Index is range 0 .. Max_Sources;

   type Extraction_Source is record
      Method     : Extraction_Method;
      Extraction : Unified_Extraction (Function_Count => 0);
      Weight     : Confidence_Level;  --  Trust weight for this source
   end record;

   type Source_Array is array (Source_Index range <>) of Extraction_Source;

   type Multi_Source_Extraction (Source_Count : Source_Index) is record
      Sources        : Source_Array (1 .. Source_Count);
      Merged_Result  : Unified_Extraction (Function_Count => 0);
   end record;

   --  ========================================================================
   --  Merger Operations
   --  ========================================================================

   procedure Add_Extraction_Source
     (Multi      : in out Multi_Source_Extraction;
      Method     : in     Extraction_Method;
      Extraction : in     Unified_Extraction;
      Weight     : in     Confidence_Level := 50;
      Status     :    out Status_Code)
   with
      Pre  => Multi.Source_Count < Max_Sources,
      Post => Multi.Source_Count <= Multi.Source_Count'Old + 1;

   procedure Merge_Extractions
     (Multi      : in out Multi_Source_Extraction;
      Merged     :    out Unified_Extraction;
      Status     :    out Status_Code)
   with
      Pre  => Multi.Source_Count > 0,
      Post => Merged.Function_Count <= Max_Functions;

   --  ========================================================================
   --  Conflict Resolution
   --  ========================================================================

   type Conflict_Resolution_Strategy is (
      Strategy_Highest_Confidence,  --  Pick source with highest confidence
      Strategy_Majority_Vote,       --  Pick type agreed by majority
      Strategy_Union_Types,         --  Combine type information
      Strategy_Most_Specific,       --  Pick most specific type
      Strategy_Reject_Conflict      --  Mark as conflict, require manual review
   );

   procedure Resolve_Type_Conflict
     (Type_Options : in     Type_Source_Array;
      Strategy     : in     Conflict_Resolution_Strategy;
      Resolved     :    out Type_Name_String;
      Confidence   :    out Confidence_Level;
      Status       :    out Status_Code);

   --  ========================================================================
   --  Confidence Calculation
   --  ========================================================================

   function Calculate_Function_Confidence
     (Function_Sig : Function_Signature;
      Sources      : Source_Array) return Confidence_Level
   with
      Pre => Sources'Length > 0;

   function Calculate_Type_Confidence
     (Type_Name    : Type_Name_String;
      Sources      : Source_Array) return Confidence_Level
   with
      Pre => Sources'Length > 0;

   --  ========================================================================
   --  Universal Parser Entry Point
   --  ========================================================================

   procedure Universal_Extract
     (Source_Code    : in     JSON_String;  --  Source code to parse
      Output_Format  :    out Unified_Extraction;
      Status         :    out Status_Code)
   with
      Post => Output_Format.Function_Count <= Max_Functions;
   --  This is the main entry point that:
   --  1. Runs multiple extraction methods on the source
   --  2. Merges the results
   --  3. Returns unified extraction with confidence scores

end Extraction_Merger;