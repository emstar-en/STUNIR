--  STUNIR DO-333 Interface Specification
--  Formal Methods Tool Integration
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides the interface to DO-333 formal methods:
--  - GNATprove integration
--  - Proof obligation collection
--  - Verification condition tracking
--  - Evidence generation

pragma SPARK_Mode (On);

with DO333_Types; use DO333_Types;

package DO333_Interface is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Path_Length        : constant := 512;
   Default_Timeout_Sec    : constant := 60;
   Default_Proof_Level    : constant := 2;

   --  ============================================================
   --  Path Types
   --  ============================================================

   subtype Path_Index is Positive range 1 .. Max_Path_Length;
   subtype Path_Length is Natural range 0 .. Max_Path_Length;
   subtype Path_String is String (Path_Index);

   --  ============================================================
   --  Verification Configuration
   --  ============================================================

   type Verify_Config is record
      Source_Dir   : Path_String;
      Source_Len   : Path_Length;
      Output_Dir   : Path_String;
      Output_Len   : Path_Length;
      Project_File : Path_String;
      Project_Len  : Path_Length;
      Timeout_Sec  : Positive;
      Proof_Level  : Positive;
      Mode         : Natural;  --  0=check, 1=flow, 2=prove
      Parallel     : Positive;
      Warnings     : Boolean;
   end record;

   Null_Verify_Config : constant Verify_Config := (
      Source_Dir   => (others => ' '),
      Source_Len   => 0,
      Output_Dir   => (others => ' '),
      Output_Len   => 0,
      Project_File => (others => ' '),
      Project_Len  => 0,
      Timeout_Sec  => Default_Timeout_Sec,
      Proof_Level  => Default_Proof_Level,
      Mode         => 2,
      Parallel     => 1,
      Warnings     => True
   );

   --  ============================================================
   --  Verification Operations
   --  ============================================================

   --  Initialize verification configuration
   procedure Initialize_Config
     (Config      : out Verify_Config;
      Source_Dir  : String;
      Output_Dir  : String;
      Project_File: String := "");

   --  Run GNATprove verification
   procedure Run_Verification
     (Config : Verify_Config;
      Result : out DO333_Result;
      Status : out DO333_Status)
   with Pre  => Config.Source_Len > 0,
        Post => (if Status = Success then Result.Success);

   --  Collect proof results from output
   procedure Collect_Proof_Results
     (Output_Dir : String;
      Result     : in out DO333_Result;
      Status     : out DO333_Status)
   with Pre => Output_Dir'Length > 0 and Output_Dir'Length <= Max_Path_Length;

   --  Generate evidence artifacts
   procedure Generate_Evidence
     (Result     : DO333_Result;
      Output_Dir : String;
      Status     : out DO333_Status)
   with Pre => Output_Dir'Length > 0 and Output_Dir'Length <= Max_Path_Length;

   --  ============================================================
   --  VC Operations
   --  ============================================================

   --  Add verification condition
   procedure Add_VC
     (Result : in out DO333_Result;
      Name   : String;
      Source : String;
      Line   : Positive;
      Column : Positive;
      Kind   : VC_Kind;
      Status : Proof_Status;
      St_Out : out DO333_Status)
   with Pre  => Name'Length > 0 and Name'Length <= Max_VC_Name_Length and
                Source'Length > 0 and Source'Length <= Max_Source_Path_Length and
                Result.VC_Total < Max_VC_Count,
        Post => (if St_Out = Success then Result.VC_Total = Result.VC_Total'Old + 1);

   --  Add proof obligation
   procedure Add_PO
     (Result  : in out DO333_Result;
      Name    : String;
      Source  : String;
      VC_Cnt  : VC_Count_Type;
      Proven  : VC_Count_Type;
      St_Out  : out DO333_Status)
   with Pre  => Name'Length > 0 and Name'Length <= Max_VC_Name_Length and
                Source'Length > 0 and Source'Length <= Max_Source_Path_Length and
                Result.PO_Total < Max_PO_Count,
        Post => (if St_Out = Success then Result.PO_Total = Result.PO_Total'Old + 1);

   --  ============================================================
   --  Validation Operations
   --  ============================================================

   --  Check if all VCs are proven
   function All_VCs_Proven (Result : DO333_Result) return Boolean;

   --  Calculate proof rate
   function Calculate_Proof_Rate (Result : DO333_Result) return Percentage_Type;

   --  Check if result meets requirements
   function Meets_Requirements
     (Result   : DO333_Result;
      Min_Rate : Percentage_Type) return Boolean;

   --  ============================================================
   --  Summary Operations
   --  ============================================================

   --  Finalize result with computed metrics
   procedure Finalize_Result
     (Result : in out DO333_Result)
   with Post => Result.Proof_Rate >= 0.0 and Result.Proof_Rate <= 100.0;

   --  Update VC statistics from array
   procedure Update_VC_Statistics
     (Result : in out DO333_Result);

end DO333_Interface;
