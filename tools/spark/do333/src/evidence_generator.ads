--  STUNIR DO-333 Certification Evidence Generator
--  Generates DO-333 compliance evidence
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides certification evidence generation:
--  - PO reports
--  - VC reports
--  - Coverage reports
--  - Compliance matrix
--  - Justification templates
--
--  DO-333 Objective: FM.6 (Certification Evidence)

pragma SPARK_Mode (On);

with PO_Manager; use PO_Manager;
with VC_Tracker; use VC_Tracker;
with Proof_Obligation; use Proof_Obligation;

package Evidence_Generator is

   --  ============================================================
   --  Output Format
   --  ============================================================

   type Output_Format is (
      Format_Text,
      Format_HTML,
      Format_JSON,
      Format_XML,
      Format_CSV
   );

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Report_Size        : constant := 1_000_000;  --  1MB
   Max_Filename_Length    : constant := 512;
   Max_Compliance_Entries : constant := 100;

   --  ============================================================
   --  Compliance Entry
   --  ============================================================

   type Compliance_Status is (
      Status_Compliant,
      Status_Partial,
      Status_Non_Compliant,
      Status_Not_Applicable
   );

   Max_Objective_Len : constant := 16;
   Max_Desc_Len      : constant := 256;
   Max_Evidence_Len  : constant := 128;
   Max_Comment_Len   : constant := 512;

   subtype Objective_String is String (1 .. Max_Objective_Len);
   subtype Desc_String is String (1 .. Max_Desc_Len);
   subtype Evidence_String is String (1 .. Max_Evidence_Len);
   subtype Comment_String is String (1 .. Max_Comment_Len);

   type Compliance_Entry is record
      Objective_ID  : Objective_String;
      Obj_Len       : Natural;
      Description   : Desc_String;
      Desc_Len      : Natural;
      Status        : Compliance_Status;
      Evidence_Ref  : Evidence_String;
      Evid_Len      : Natural;
      Comments      : Comment_String;
      Comment_Len   : Natural;
   end record;

   Empty_Compliance_Entry : constant Compliance_Entry := (
      Objective_ID => (others => ' '),
      Obj_Len      => 0,
      Description  => (others => ' '),
      Desc_Len     => 0,
      Status       => Status_Not_Applicable,
      Evidence_Ref => (others => ' '),
      Evid_Len     => 0,
      Comments     => (others => ' '),
      Comment_Len  => 0
   );

   --  ============================================================
   --  Compliance Matrix
   --  ============================================================

   subtype Compliance_Index is Positive range 1 .. Max_Compliance_Entries;
   subtype Compliance_Count is Natural range 0 .. Max_Compliance_Entries;

   type Compliance_Array is array (Compliance_Index) of Compliance_Entry;

   type Compliance_Matrix is record
      Entries : Compliance_Array;
      Count   : Compliance_Count;
   end record;

   Empty_Matrix : constant Compliance_Matrix := (
      Entries => (others => Empty_Compliance_Entry),
      Count   => 0
   );

   --  ============================================================
   --  Report Generation Result
   --  ============================================================

   type Generation_Result is (
      Gen_Success,
      Gen_Error_IO,
      Gen_Error_Overflow,
      Gen_Error_Format
   );

   --  ============================================================
   --  Report Statistics
   --  ============================================================

   type Report_Statistics is record
      Total_Lines     : Natural;
      Total_Bytes     : Natural;
      Generation_Time : Natural;  --  milliseconds
   end record;

   Empty_Statistics : constant Report_Statistics := (
      Total_Lines     => 0,
      Total_Bytes     => 0,
      Generation_Time => 0
   );

   --  ============================================================
   --  Report Generation Operations
   --  ============================================================

   --  Generate PO report
   procedure Generate_PO_Report
     (PO_Coll  : PO_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   with
      Pre => Content'Length >= Max_Report_Size;

   --  Generate VC report
   procedure Generate_VC_Report
     (VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   with
      Pre => Content'Length >= Max_Report_Size;

   --  Generate coverage report
   procedure Generate_Coverage_Report
     (PO_Coll  : PO_Collection;
      VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   with
      Pre => Content'Length >= Max_Report_Size;

   --  ============================================================
   --  Compliance Matrix Operations
   --  ============================================================

   --  Initialize compliance matrix with DO-333 objectives
   procedure Initialize_Matrix
     (Matrix : out Compliance_Matrix)
   with
      Post => Matrix.Count > 0;

   --  Update matrix entry
   procedure Update_Matrix_Entry
     (Matrix      : in Out Compliance_Matrix;
      Objective   : String;
      Status      : Compliance_Status;
      Evidence    : String;
      Comment     : String);

   --  Generate compliance matrix report
   procedure Generate_Compliance_Matrix
     (PO_Coll  : PO_Collection;
      VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   with
      Pre => Content'Length >= Max_Report_Size;

   --  ============================================================
   --  Justification Operations
   --  ============================================================

   --  Generate justification template for unproven POs
   procedure Generate_Justification_Template
     (PO_Coll  : PO_Collection;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   with
      Pre => Content'Length >= Max_Report_Size;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Get format name as string
   function Format_Name (F : Output_Format) return String;

   --  Get file extension for format
   function Format_Extension (F : Output_Format) return String;

   --  Get compliance status name
   function Status_Name (S : Compliance_Status) return String;

end Evidence_Generator;
