--  STUNIR DO-333 Specification Parser
--  Parse formal specifications from Ada/SPARK source
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package parses formal specifications:
--  - SPARK Pre/Post contracts
--  - Loop invariants
--  - Type invariants
--  - Ghost code
--
--  DO-333 Objective: FM.1 (Formal Specification)

pragma SPARK_Mode (On);

with Formal_Spec; use Formal_Spec;

package Spec_Parser is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Source_Size : constant := 1_000_000;  --  1MB max source
   Max_Filename    : constant := 512;

   --  ============================================================
   --  Parse Result
   --  ============================================================

   type Parse_Result is (
      Parse_Success,
      Parse_Error_Syntax,
      Parse_Error_Overflow,
      Parse_Error_IO,
      Parse_Error_Empty
   );

   --  ============================================================
   --  Source Location
   --  ============================================================

   type Source_Location is record
      Line   : Natural;
      Column : Natural;
      Offset : Natural;
   end record;

   --  ============================================================
   --  Parser State
   --  ============================================================

   type Parser_State is record
      Current_Line   : Natural;
      Current_Column : Natural;
      Position       : Natural;
      In_Contract    : Boolean;
      In_Ghost       : Boolean;
      Error_Count    : Natural;
   end record;

   Initial_State : constant Parser_State := (
      Current_Line   => 1,
      Current_Column => 1,
      Position       => 1,
      In_Contract    => False,
      In_Ghost       => False,
      Error_Count    => 0
   );

   --  ============================================================
   --  Parse Statistics
   --  ============================================================

   type Parse_Statistics is record
      Total_Lines      : Natural;
      Contract_Lines   : Natural;
      Pre_Count        : Natural;
      Post_Count       : Natural;
      Invariant_Count  : Natural;
      Ghost_Count      : Natural;
      Parse_Time_MS    : Natural;
   end record;

   Empty_Statistics : constant Parse_Statistics := (
      Total_Lines     => 0,
      Contract_Lines  => 0,
      Pre_Count       => 0,
      Post_Count      => 0,
      Invariant_Count => 0,
      Ghost_Count     => 0,
      Parse_Time_MS   => 0
   );

   --  ============================================================
   --  Operations
   --  ============================================================

   --  Parse contracts from source content string
   procedure Parse_Source_Content
     (Source    : String;
      Contracts : out Contract_Spec;
      Stats     : out Parse_Statistics;
      Result    : out Parse_Result)
   with
      Pre  => Source'Length > 0 and then Source'Length <= Max_Source_Size,
      Post => (Result = Parse_Success) = Is_Valid_Contract (Contracts);

   --  Parse a single expression (e.g., from IR)
   procedure Parse_Single_Expression
     (Expr_Text : String;
      Kind      : Spec_Kind;
      Expr      : out Formal_Expression;
      Result    : out Parse_Result)
   with
      Pre => Expr_Text'Length > 0 and then 
             Expr_Text'Length <= Max_Expression_Length;

   --  Check if line contains Pre aspect
   function Contains_Pre (Line : String) return Boolean
   with
      Pre => Line'Length > 0;

   --  Check if line contains Post aspect
   function Contains_Post (Line : String) return Boolean
   with
      Pre => Line'Length > 0;

   --  Check if line contains invariant
   function Contains_Invariant (Line : String) return Boolean
   with
      Pre => Line'Length > 0;

   --  Check if line is ghost code
   function Is_Ghost_Line (Line : String) return Boolean
   with
      Pre => Line'Length > 0;

end Spec_Parser;
