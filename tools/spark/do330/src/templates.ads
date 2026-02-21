--  STUNIR DO-330 Template Types and Operations
--  Template System for Tool Qualification Documents
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides template types for DO-330 artifacts:
--  - TOR (Tool Operational Requirements)
--  - TQP (Tool Qualification Plan)
--  - TAS (Tool Accomplishment Summary)
--  - VCP (Verification Cases and Procedures)
--  - CI (Configuration Index)
--
--  DO-330 Objective: T-0 through T-5 (Tool Qualification)

pragma SPARK_Mode (On);

package Templates is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Name_Length      : constant := 256;
   Max_Value_Length     : constant := 4096;
   Max_Path_Length      : constant := 1024;
   Max_Template_Size    : constant := 65536;
   Max_Variables        : constant := 64;
   Max_Requirements     : constant := 256;
   Max_Test_Cases       : constant := 512;

   --  ============================================================
   --  Template Kind Enumeration
   --  ============================================================

   type Template_Kind is (
      TOR_Template,           --  Tool Operational Requirements
      TQP_Template,           --  Tool Qualification Plan
      TAS_Template,           --  Tool Accomplishment Summary
      VCP_Template,           --  Verification Cases and Procedures
      CI_Template,            --  Configuration Index
      Traceability_Template,  --  Traceability Matrix
      Problem_Report_Template --  Problem Report
   );

   --  ============================================================
   --  Tool Qualification Level (DO-330)
   --  ============================================================

   type TQL_Level is (
      TQL_1,  --  Most rigorous (DAL A output, Criteria 1/2)
      TQL_2,  --  High rigor (DAL B output)
      TQL_3,  --  Moderate rigor (DAL C output)
      TQL_4,  --  Lower rigor (Verification tools)
      TQL_5   --  No qualification required
   );

   --  ============================================================
   --  Design Assurance Level (DO-178C/278A)
   --  ============================================================

   type DAL_Level is (
      DAL_A,  --  Catastrophic failure condition
      DAL_B,  --  Hazardous failure condition
      DAL_C,  --  Major failure condition
      DAL_D,  --  Minor failure condition
      DAL_E   --  No safety effect
   );

   --  ============================================================
   --  Verification Method
   --  ============================================================

   type Verification_Method is (
      Test,      --  Verification by testing
      Analysis,  --  Verification by analysis
      Review,    --  Verification by review
      Formal     --  Formal verification (DO-333)
   );

   --  ============================================================
   --  Bounded Strings
   --  ============================================================

   subtype Name_Index is Positive range 1 .. Max_Name_Length;
   subtype Name_Length_Type is Natural range 0 .. Max_Name_Length;
   subtype Name_String is String (Name_Index);

   subtype Value_Index is Positive range 1 .. Max_Value_Length;
   subtype Value_Length_Type is Natural range 0 .. Max_Value_Length;
   subtype Value_String is String (Value_Index);

   subtype Path_Index is Positive range 1 .. Max_Path_Length;
   subtype Path_Length_Type is Natural range 0 .. Max_Path_Length;
   subtype Path_String is String (Path_Index);

   --  ============================================================
   --  Template Variable
   --  ============================================================

   type Template_Variable is record
      Name        : Name_String;
      Name_Len    : Name_Length_Type;
      Value       : Value_String;
      Value_Len   : Value_Length_Type;
      Is_Set      : Boolean;
   end record;

   --  Default values for Template_Variable
   Null_Template_Variable : constant Template_Variable := (
      Name      => (others => ' '),
      Name_Len  => 0,
      Value     => (others => ' '),
      Value_Len => 0,
      Is_Set    => False
   );

   --  ============================================================
   --  Tool Requirement (TOR)
   --  ============================================================

   subtype Requirement_ID_Length is Natural range 0 .. 32;
   subtype Requirement_ID_String is String (1 .. 32);

   type Requirement_Category is (
      Functional,     --  Functional requirements
      Environmental,  --  Environmental requirements
      Interface_Req,  --  Interface requirements
      Constraint,     --  Constraints
      Performance     --  Performance requirements
   );

   type Tool_Requirement is record
      ID          : Requirement_ID_String;
      ID_Len      : Requirement_ID_Length;
      Category    : Requirement_Category;
      Description : Value_String;
      Desc_Len    : Value_Length_Type;
      Method      : Verification_Method;
      Is_Valid    : Boolean;
   end record;

   Null_Tool_Requirement : constant Tool_Requirement := (
      ID          => (others => ' '),
      ID_Len      => 0,
      Category    => Functional,
      Description => (others => ' '),
      Desc_Len    => 0,
      Method      => Test,
      Is_Valid    => False
   );

   --  ============================================================
   --  Test Case Reference
   --  ============================================================

   type Test_Status is (
      Not_Run,
      Passed,
      Failed,
      Blocked,
      Skipped
   );

   type Test_Case is record
      ID          : Requirement_ID_String;
      ID_Len      : Requirement_ID_Length;
      TOR_Ref     : Requirement_ID_String;
      TOR_Ref_Len : Requirement_ID_Length;
      Description : Value_String;
      Desc_Len    : Value_Length_Type;
      Status      : Test_Status;
      Is_Valid    : Boolean;
   end record;

   Null_Test_Case : constant Test_Case := (
      ID          => (others => ' '),
      ID_Len      => 0,
      TOR_Ref     => (others => ' '),
      TOR_Ref_Len => 0,
      Description => (others => ' '),
      Desc_Len    => 0,
      Status      => Not_Run,
      Is_Valid    => False
   );

   --  ============================================================
   --  Variable Array Type
   --  ============================================================

   subtype Variable_Index is Positive range 1 .. Max_Variables;
   subtype Variable_Count is Natural range 0 .. Max_Variables;

   type Variable_Array is array (Variable_Index) of Template_Variable;

   --  ============================================================
   --  Requirements Array Type
   --  ============================================================

   subtype Requirement_Index is Positive range 1 .. Max_Requirements;
   subtype Requirement_Count is Natural range 0 .. Max_Requirements;

   type Requirement_Array is array (Requirement_Index) of Tool_Requirement;

   --  ============================================================
   --  Test Cases Array Type
   --  ============================================================

   subtype Test_Case_Index is Positive range 1 .. Max_Test_Cases;
   subtype Test_Case_Count is Natural range 0 .. Max_Test_Cases;

   type Test_Case_Array is array (Test_Case_Index) of Test_Case;

   --  ============================================================
   --  Utility Functions
   --  ============================================================

   --  Convert TQL level to string
   function TQL_To_String (Level : TQL_Level) return String
   with Post => TQL_To_String'Result'Length <= 6;

   --  Convert DAL level to string
   function DAL_To_String (Level : DAL_Level) return String
   with Post => DAL_To_String'Result'Length <= 6;

   --  Convert verification method to string
   function Method_To_String (Method : Verification_Method) return String
   with Post => Method_To_String'Result'Length <= 8;

   --  Convert test status to string
   function Status_To_String (Status : Test_Status) return String
   with Post => Status_To_String'Result'Length <= 8;

   --  Convert template kind to filename
   function Kind_To_Filename (Kind : Template_Kind) return String
   with Post => Kind_To_Filename'Result'Length <= 32;

   --  Validate requirement ID format
   function Is_Valid_Requirement_ID (ID : String) return Boolean
   with Pre => ID'Length <= 32;

   --  Validate test case ID format
   function Is_Valid_Test_Case_ID (ID : String) return Boolean
   with Pre => ID'Length <= 32;

end Templates;
