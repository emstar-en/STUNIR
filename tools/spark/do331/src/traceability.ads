--  STUNIR DO-331 Traceability Framework Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides bidirectional traceability tracking between
--  IR elements and model elements for DO-331 compliance.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;

package Traceability is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Trace_Entries : constant := 50_000;
   Max_Path_Length   : constant := 512;

   --  ============================================================
   --  Trace Types
   --  ============================================================

   type Trace_Type is (
      Trace_IR_To_Model,        --  Forward: IR -> Model
      Trace_Model_To_IR,        --  Backward: Model -> IR
      Trace_Model_To_Model,     --  Internal model trace
      Trace_Model_To_Req,       --  Model -> Requirement
      Trace_Req_To_Test         --  Requirement -> Test
   );

   --  ============================================================
   --  Trace Entry
   --  ============================================================

   type Trace_Entry is record
      Trace_ID        : Natural;
      Trace_Kind      : Trace_Type;
      Source_ID       : Element_ID;
      Source_Path     : String (1 .. Max_Path_Length);
      Source_Path_Len : Natural;
      Target_ID       : Element_ID;
      Target_Path     : String (1 .. Max_Path_Length);
      Target_Path_Len : Natural;
      Rule            : Transformation_Rule;
      Timestamp       : Natural;  --  Epoch seconds
      Verified        : Boolean;
   end record;

   Null_Trace_Entry : constant Trace_Entry := (
      Trace_ID        => 0,
      Trace_Kind      => Trace_IR_To_Model,
      Source_ID       => Null_Element_ID,
      Source_Path     => (others => ' '),
      Source_Path_Len => 0,
      Target_ID       => Null_Element_ID,
      Target_Path     => (others => ' '),
      Target_Path_Len => 0,
      Rule            => Rule_Module_To_Package,
      Timestamp       => 0,
      Verified        => False
   );

   --  ============================================================
   --  Trace Entry Array
   --  ============================================================

   type Trace_Entry_Array is array (Positive range <>) of Trace_Entry;

   --  ============================================================
   --  Trace Matrix
   --  ============================================================

   type Trace_Matrix is record
      Entries      : Trace_Entry_Array (1 .. Max_Trace_Entries);
      Entry_Count  : Natural := 0;
      IR_Hash      : String (1 .. 64);
      IR_Hash_Len  : Natural := 0;
      Model_Hash   : String (1 .. 64);
      Model_Hash_Len : Natural := 0;
      Created_At   : Natural := 0;
   end record;

   --  ============================================================
   --  Gap Report
   --  ============================================================

   type Gap_Report is record
      Total_IR_Elements   : Natural;
      Traced_Elements     : Natural;
      Missing_Traces      : Natural;
      Gap_Percentage      : Natural;  --  0-100
      Is_Complete         : Boolean;
   end record;

   Null_Gap_Report : constant Gap_Report := (
      Total_IR_Elements => 0,
      Traced_Elements   => 0,
      Missing_Traces    => 0,
      Gap_Percentage    => 100,
      Is_Complete       => False
   );

   --  ============================================================
   --  Matrix Operations
   --  ============================================================

   --  Create empty trace matrix
   function Create_Matrix return Trace_Matrix
     with Post => Create_Matrix'Result.Entry_Count = 0;

   --  Add a trace entry
   procedure Add_Trace (
      Matrix    : in Out Trace_Matrix;
      Kind      : in     Trace_Type;
      Source_ID : in     Element_ID;
      Src_Path  : in     String;
      Target_ID : in     Element_ID;
      Tgt_Path  : in     String;
      Rule      : in     Transformation_Rule
   ) with
      Pre  => Matrix.Entry_Count < Max_Trace_Entries,
      Post => Matrix.Entry_Count = Matrix.Entry_Count'Old + 1;

   --  Set IR hash
   procedure Set_IR_Hash (
      Matrix : in Out Trace_Matrix;
      Hash   : in     String
   ) with Pre => Hash'Length > 0 and Hash'Length <= 64;

   --  Set model hash
   procedure Set_Model_Hash (
      Matrix : in Out Trace_Matrix;
      Hash   : in     String
   ) with Pre => Hash'Length > 0 and Hash'Length <= 64;

   --  ============================================================
   --  Lookup Operations
   --  ============================================================

   --  Get traces by source element
   function Get_Forward_Traces (
      Matrix    : Trace_Matrix;
      Source_ID : Element_ID
   ) return Trace_Entry_Array;

   --  Get traces by target element
   function Get_Backward_Traces (
      Matrix    : Trace_Matrix;
      Target_ID : Element_ID
   ) return Trace_Entry_Array;

   --  Get traces by rule
   function Get_Traces_By_Rule (
      Matrix : Trace_Matrix;
      Rule   : Transformation_Rule
   ) return Trace_Entry_Array;

   --  Check if element has trace
   function Has_Trace (
      Matrix : Trace_Matrix;
      ID     : Element_ID
   ) return Boolean;

   --  ============================================================
   --  Completeness Analysis
   --  ============================================================

   --  Check completeness against IR elements
   function Check_Completeness (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Boolean;

   --  Analyze gaps in traceability
   function Analyze_Gaps (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Gap_Report;

   --  Get missing traces
   function Get_Missing_Traces (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Element_ID_Array;

   --  ============================================================
   --  Verification
   --  ============================================================

   --  Mark trace as verified
   procedure Mark_Verified (
      Matrix   : in Out Trace_Matrix;
      Trace_ID : in     Natural
   );

   --  Get verification status
   function Get_Verification_Status (
      Matrix : Trace_Matrix
   ) return Natural;  --  Percentage verified (0-100)

   --  ============================================================
   --  Validation
   --  ============================================================

   --  Validate matrix integrity
   function Validate_Matrix (Matrix : Trace_Matrix) return Boolean;

end Traceability;
