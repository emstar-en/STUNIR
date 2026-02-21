--  STUNIR DO-333 SPARK Integration
--  GNATprove integration for formal verification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides SPARK/GNATprove integration:
--  - Proof mode configuration (flow, prove, all)
--  - Prover selection (CVC4, Z3, Alt-Ergo, CVC5)
--  - Timeout and step configuration
--  - Proof replay support
--
--  DO-333 Objectives: FM.2, FM.5 (Proofs, Integration)

pragma SPARK_Mode (On);

package SPARK_Integration is

   --  ============================================================
   --  Proof Modes
   --  ============================================================

   type Proof_Mode is (
      Mode_Check,   --  Check SPARK compliance only
      Mode_Flow,    --  Flow analysis only
      Mode_Prove,   --  Proofs only (no flow)
      Mode_All,     --  Both flow and proofs
      Mode_Gold     --  Gold level (maximum rigor)
   );

   --  ============================================================
   --  Prover Selection
   --  ============================================================

   type Prover_Kind is (
      Prover_CVC4,
      Prover_Z3,
      Prover_Alt_Ergo,
      Prover_CVC5,
      Prover_Colibri,
      Prover_All      --  Try all provers
   );

   --  ============================================================
   --  Proof Level (effort)
   --  ============================================================

   type Proof_Level is range 0 .. 4;
   --  0 = fastest, least effort (basic assertions only)
   --  1 = fast, low effort
   --  2 = moderate effort (recommended default)
   --  3 = high effort
   --  4 = maximum effort (slowest, most thorough)

   --  ============================================================
   --  SPARK Configuration
   --  ============================================================

   type SPARK_Config is record
      Mode        : Proof_Mode;
      Prover      : Prover_Kind;
      Level       : Proof_Level;
      Timeout     : Natural;    --  seconds per VC
      Steps       : Natural;    --  max steps per VC
      Parallel    : Natural;    --  parallel jobs (0 = auto)
      Replay      : Boolean;    --  proof replay mode
      Force       : Boolean;    --  force reproof of all VCs
      Warnings    : Boolean;    --  show warnings
      Debug       : Boolean;    --  debug output
      Output_JSON : Boolean;    --  JSON output format
   end record;

   --  Default configuration
   Default_Config : constant SPARK_Config := (
      Mode        => Mode_All,
      Prover      => Prover_All,
      Level       => 2,
      Timeout     => 60,
      Steps       => 10000,
      Parallel    => 0,       --  Auto-detect
      Replay      => False,
      Force       => False,
      Warnings    => True,
      Debug       => False,
      Output_JSON => True
   );

   --  High-assurance configuration (DAL-A/B)
   High_Assurance_Config : constant SPARK_Config := (
      Mode        => Mode_All,
      Prover      => Prover_All,
      Level       => 4,
      Timeout     => 120,
      Steps       => 50000,
      Parallel    => 0,
      Replay      => True,
      Force       => False,
      Warnings    => True,
      Debug       => False,
      Output_JSON => True
   );

   --  Quick check configuration
   Quick_Config : constant SPARK_Config := (
      Mode        => Mode_Flow,
      Prover      => Prover_Z3,
      Level       => 0,
      Timeout     => 10,
      Steps       => 1000,
      Parallel    => 0,
      Replay      => False,
      Force       => False,
      Warnings    => False,
      Debug       => False,
      Output_JSON => True
   );

   --  ============================================================
   --  SPARK Result
   --  ============================================================

   type SPARK_Result is record
      Success       : Boolean;
      Exit_Code     : Integer;
      Total_VCs     : Natural;
      Proved_VCs    : Natural;
      Unproved_VCs  : Natural;
      Flow_Errors   : Natural;
      Flow_Warnings : Natural;
      Elapsed_Time  : Natural;  --  seconds
      Tool_Version  : Natural;  --  GNATprove version number
   end record;

   --  Empty result constant
   Empty_Result : constant SPARK_Result := (
      Success       => False,
      Exit_Code     => -1,
      Total_VCs     => 0,
      Proved_VCs    => 0,
      Unproved_VCs  => 0,
      Flow_Errors   => 0,
      Flow_Warnings => 0,
      Elapsed_Time  => 0,
      Tool_Version  => 0
   );

   --  ============================================================
   --  Tool Availability
   --  ============================================================

   --  Check if GNATprove is available
   function Is_GNATprove_Available return Boolean;

   --  Get GNATprove version string
   function Get_GNATprove_Version return String;

   --  Check if specific prover is available
   function Is_Prover_Available (P : Prover_Kind) return Boolean;

   --  ============================================================
   --  Configuration Operations
   --  ============================================================

   --  Get mode name as string
   function Mode_Name (M : Proof_Mode) return String;

   --  Get prover name as string
   function Prover_Name (P : Prover_Kind) return String;

   --  Validate configuration
   function Is_Valid_Config (C : SPARK_Config) return Boolean is
     (C.Timeout > 0 and then C.Steps > 0);

   --  ============================================================
   --  Result Analysis
   --  ============================================================

   --  Check if all VCs proved
   function All_Proved (R : SPARK_Result) return Boolean is
     (R.Success and then R.Unproved_VCs = 0 and then R.Flow_Errors = 0);

   --  Get proof percentage
   function Proof_Percentage (R : SPARK_Result) return Natural is
     (if R.Total_VCs > 0 then (R.Proved_VCs * 100) / R.Total_VCs else 0);

   --  Check if flow analysis passed
   function Flow_Passed (R : SPARK_Result) return Boolean is
     (R.Flow_Errors = 0);

end SPARK_Integration;
