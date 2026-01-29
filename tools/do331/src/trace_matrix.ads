--  STUNIR DO-331 Trace Matrix Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides trace matrix export functionality for DO-331.

pragma SPARK_Mode (On);

with Traceability; use Traceability;
with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;

package Trace_Matrix is

   --  ============================================================
   --  Export Format
   --  ============================================================

   type Export_Format is (
      Format_JSON,
      Format_CSV,
      Format_Text
   );

   --  ============================================================
   --  Export Buffer
   --  ============================================================

   Max_Export_Length : constant := 500_000;  --  500 KB

   type Export_Buffer is record
      Data   : String (1 .. Max_Export_Length);
      Length : Natural := 0;
   end record;

   --  Initialize buffer
   procedure Initialize_Export_Buffer (Buffer : out Export_Buffer);

   --  Append to buffer
   procedure Append_Export (
      Buffer : in Out Export_Buffer;
      Text   : in     String
   );

   --  Get buffer content
   function Get_Export_Content (Buffer : Export_Buffer) return String
     with Pre => Buffer.Length <= Max_Export_Length;

   --  ============================================================
   --  Export Operations
   --  ============================================================

   --  Export trace matrix to JSON format
   procedure Export_To_JSON (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   );

   --  Export trace matrix to CSV format
   procedure Export_To_CSV (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   );

   --  Export trace matrix to text format
   procedure Export_To_Text (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   );

   --  ============================================================
   --  Summary Generation
   --  ============================================================

   type Matrix_Summary is record
      Total_Entries     : Natural;
      Forward_Traces    : Natural;
      Backward_Traces   : Natural;
      Verified_Count    : Natural;
      Unique_Sources    : Natural;
      Unique_Targets    : Natural;
      Rules_Used        : Natural;
   end record;

   --  Generate summary of trace matrix
   function Get_Summary (
      Matrix : Traceability.Trace_Matrix
   ) return Matrix_Summary;

   --  ============================================================
   --  DO-331 Compliance Helpers
   --  ============================================================

   --  Get trace entries for DO-331 Table MB-1 format
   procedure Export_DO331_Table_MB1 (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   );

   --  Get bidirectional trace report
   procedure Export_Bidirectional_Report (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   );

end Trace_Matrix;
