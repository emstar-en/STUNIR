--  STUNIR DO-333 Proof Obligation Types
--  Proof Obligation management for formal verification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides proof obligation types supporting:
--  - PO status tracking (proved, unproved, timeout, etc.)
--  - PO classification by kind (precondition, range check, etc.)
--  - Criticality levels (DAL-A through DAL-E)
--  - Discharge strategies
--
--  DO-333 Objectives: FM.2, FM.3 (Proofs, Coverage)

pragma SPARK_Mode (On);

package Proof_Obligation is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_PO_Count      : constant := 10000;
   Max_Source_Len    : constant := 512;
   Max_Context_Len   : constant := 2048;
   Max_Subp_Name_Len : constant := 256;

   --  ============================================================
   --  PO Status
   --  ============================================================

   type PO_Status is (
      PO_Unproved,
      PO_Proved,
      PO_Timeout,
      PO_Error,
      PO_Manually_Justified,
      PO_Flow_Proved,
      PO_Skipped
   );

   --  ============================================================
   --  PO Kind (matches GNATprove categories)
   --  ============================================================

   type PO_Kind is (
      PO_Precondition,
      PO_Postcondition,
      PO_Loop_Invariant_Init,
      PO_Loop_Invariant_Preserv,
      PO_Loop_Variant,
      PO_Assert,
      PO_Range_Check,
      PO_Overflow_Check,
      PO_Division_Check,
      PO_Index_Check,
      PO_Discriminant_Check,
      PO_Predicate_Check,
      PO_Default_Initial_Condition,
      PO_Contract_Cases,
      PO_Initial_Condition,
      PO_Type_Invariant,
      PO_Tag_Check,
      PO_Ceiling_Priority,
      PO_Subprogram_Variant,
      PO_Other
   );

   --  ============================================================
   --  Criticality Level (DAL)
   --  ============================================================

   type Criticality_Level is (
      DAL_A,  --  Catastrophic
      DAL_B,  --  Hazardous
      DAL_C,  --  Major
      DAL_D,  --  Minor
      DAL_E   --  No effect
   );

   --  ============================================================
   --  Discharge Strategy
   --  ============================================================

   type Discharge_Strategy is (
      Strategy_Auto,      --  Automatic prover
      Strategy_Manual,    --  Manual proof
      Strategy_Lemma,     --  Using lemma library
      Strategy_Review,    --  Code review (non-proof)
      Strategy_Test       --  Testing (fallback)
   );

   --  ============================================================
   --  Bounded Strings
   --  ============================================================

   subtype Source_Index is Positive range 1 .. Max_Source_Len;
   subtype Source_Length is Natural range 0 .. Max_Source_Len;
   subtype Source_String is String (Source_Index);

   subtype Context_Index is Positive range 1 .. Max_Context_Len;
   subtype Context_Length is Natural range 0 .. Max_Context_Len;
   subtype Context_String is String (Context_Index);

   subtype Subp_Index is Positive range 1 .. Max_Subp_Name_Len;
   subtype Subp_Length is Natural range 0 .. Max_Subp_Name_Len;
   subtype Subp_String is String (Subp_Index);

   --  ============================================================
   --  Proof Obligation Record
   --  ============================================================

   type Proof_Obligation_Record is record
      ID           : Natural;
      Kind         : PO_Kind;
      Status       : PO_Status;
      Criticality  : Criticality_Level;
      Strategy     : Discharge_Strategy;
      Source_File  : Source_String;
      Source_Len   : Source_Length;
      Subprogram   : Subp_String;
      Subp_Len     : Subp_Length;
      Line         : Natural;
      Column       : Natural;
      Context      : Context_String;
      Context_Len  : Context_Length;
      Prover_Time  : Natural;  --  milliseconds
      Step_Count   : Natural;
      Prover_Name  : Source_String;
      Prover_Len   : Source_Length;
   end record;

   --  ============================================================
   --  Empty/Default PO
   --  ============================================================

   Empty_PO : constant Proof_Obligation_Record := (
      ID          => 0,
      Kind        => PO_Other,
      Status      => PO_Unproved,
      Criticality => DAL_E,
      Strategy    => Strategy_Auto,
      Source_File => (others => ' '),
      Source_Len  => 0,
      Subprogram  => (others => ' '),
      Subp_Len    => 0,
      Line        => 0,
      Column      => 0,
      Context     => (others => ' '),
      Context_Len => 0,
      Prover_Time => 0,
      Step_Count  => 0,
      Prover_Name => (others => ' '),
      Prover_Len  => 0
   );

   --  ============================================================
   --  Predicates
   --  ============================================================

   function Is_Valid_PO (PO : Proof_Obligation_Record) return Boolean is
     (PO.Source_Len > 0 and then PO.Line > 0);

   function Is_Proved (PO : Proof_Obligation_Record) return Boolean is
     (PO.Status in PO_Proved | PO_Flow_Proved | PO_Manually_Justified);

   function Is_Critical (PO : Proof_Obligation_Record) return Boolean is
     (PO.Criticality in DAL_A | DAL_B);

   function Is_Safety_Related (PO : Proof_Obligation_Record) return Boolean is
     (PO.Criticality in DAL_A | DAL_B | DAL_C);

   function Needs_Attention (PO : Proof_Obligation_Record) return Boolean is
     (not Is_Proved (PO) and then Is_Safety_Related (PO));

   --  ============================================================
   --  Operations
   --  ============================================================

   --  Create a new PO record
   procedure Create_PO
     (ID          : Natural;
      Kind        : PO_Kind;
      Criticality : Criticality_Level;
      Source      : String;
      Subprogram  : String;
      Line        : Natural;
      Column      : Natural;
      PO          : out Proof_Obligation_Record)
   with
      Pre  => Source'Length > 0 and then Source'Length <= Max_Source_Len and then
              Subprogram'Length <= Max_Subp_Name_Len and then
              Line > 0,
      Post => Is_Valid_PO (PO) and then PO.ID = ID;

   --  Update PO status
   procedure Update_Status
     (PO     : in Out Proof_Obligation_Record;
      Status : PO_Status;
      Time   : Natural;
      Steps  : Natural);

   --  Set prover name
   procedure Set_Prover
     (PO   : in Out Proof_Obligation_Record;
      Name : String)
   with
      Pre => Name'Length <= Max_Source_Len;

   --  Get status name as string
   function Status_Name (S : PO_Status) return String;

   --  Get kind name as string
   function Kind_Name (K : PO_Kind) return String;

   --  Get criticality name as string
   function Criticality_Name (C : Criticality_Level) return String;

end Proof_Obligation;
