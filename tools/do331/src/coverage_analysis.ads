--  STUNIR DO-331 Coverage Analysis Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides coverage analysis and reporting for DO-331.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;
with Coverage; use Coverage;

package Coverage_Analysis is

   --  ============================================================
   --  Coverage Statistics
   --  ============================================================

   type Coverage_Stats is record
      Total_Points       : Natural;
      State_Points       : Natural;
      Transition_Points  : Natural;
      Decision_Points    : Natural;
      Condition_Points   : Natural;
      MCDC_Points        : Natural;
      Action_Points      : Natural;
      Entry_Points       : Natural;
      Exit_Points        : Natural;
      Path_Points        : Natural;
      Loop_Points        : Natural;
      Instrumented_Count : Natural;
      Covered_Count      : Natural;
   end record;

   Null_Coverage_Stats : constant Coverage_Stats := (
      Total_Points       => 0,
      State_Points       => 0,
      Transition_Points  => 0,
      Decision_Points    => 0,
      Condition_Points   => 0,
      MCDC_Points        => 0,
      Action_Points      => 0,
      Entry_Points       => 0,
      Exit_Points        => 0,
      Path_Points        => 0,
      Loop_Points        => 0,
      Instrumented_Count => 0,
      Covered_Count      => 0
   );

   --  ============================================================
   --  DAL Coverage Summary
   --  ============================================================

   type DAL_Coverage_Summary is record
      DAL              : DAL_Level;
      Required_Points  : Natural;
      Achieved_Points  : Natural;
      Coverage_Percent : Natural;  --  0-100
      Meets_Objective  : Boolean;
   end record;

   type DAL_Coverage_Array is array (DAL_Level) of DAL_Coverage_Summary;

   --  ============================================================
   --  Analysis Result
   --  ============================================================

   type Analysis_Result is record
      Model_Hash      : String (1 .. 64);
      Hash_Length     : Natural;
      Stats           : Coverage_Stats;
      DAL_Coverage    : DAL_Coverage_Array;
      Analysis_Time   : Natural;
      Is_Complete     : Boolean;
      Target_DAL      : DAL_Level;
      Meets_Target    : Boolean;
   end record;

   Null_Analysis_Result : constant Analysis_Result := (
      Model_Hash    => (others => '0'),
      Hash_Length   => 0,
      Stats         => Null_Coverage_Stats,
      DAL_Coverage  => (others => (DAL => DAL_E, Required_Points => 0,
                                   Achieved_Points => 0, Coverage_Percent => 0,
                                   Meets_Objective => False)),
      Analysis_Time => 0,
      Is_Complete   => False,
      Target_DAL    => DAL_C,
      Meets_Target  => False
   );

   --  ============================================================
   --  Analysis Operations
   --  ============================================================

   --  Analyze coverage points
   function Analyze (Points : Coverage_Points) return Analysis_Result;

   --  Analyze for specific DAL level
   function Analyze_For_DAL (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return Analysis_Result;

   --  Compute statistics
   function Compute_Stats (Points : Coverage_Points) return Coverage_Stats;

   --  Check if coverage meets DAL requirements
   function Meets_DAL_Requirements (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return Boolean;

   --  ============================================================
   --  Report Generation
   --  ============================================================

   type Report_Format is (JSON_Format, Text_Format, HTML_Format, CSV_Format);

   Max_Report_Length : constant := 500_000;

   type Report_Buffer is record
      Data   : String (1 .. Max_Report_Length);
      Length : Natural := 0;
   end record;

   --  Initialize report buffer
   procedure Initialize_Report (Buffer : out Report_Buffer);

   --  Generate coverage report
   procedure Generate_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Format  : in     Report_Format;
      Buffer  : in Out Report_Buffer
   );

   --  Generate JSON report
   procedure Generate_JSON_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   );

   --  Generate text report
   procedure Generate_Text_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   );

   --  Generate HTML report
   procedure Generate_HTML_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   );

   --  Get report content
   function Get_Report_Content (Buffer : Report_Buffer) return String
     with Pre => Buffer.Length <= Max_Report_Length;

   --  ============================================================
   --  DO-331 Compliance Helpers
   --  ============================================================

   --  Get DO-331 Table MB-A.5 compliance status
   function Get_Table_MBA5_Status (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return String;

   --  Check specific coverage objective
   function Check_Coverage_Objective (
      Points     : Coverage_Points;
      Level      : DAL_Level;
      Cov_Type   : Coverage_Type
   ) return Boolean;

end Coverage_Analysis;
