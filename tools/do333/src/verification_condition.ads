--  STUNIR DO-333 Verification Condition Types
--  VC management for formal verification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides verification condition types supporting:
--  - VC status tracking
--  - Complexity analysis
--  - Manual proof support
--
--  DO-333 Objective: FM.4 (Verification Condition Management)

pragma SPARK_Mode (On);

package Verification_Condition is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_VC_Count      : constant := 50000;
   Max_Formula_Len   : constant := 8192;
   Max_Reason_Len    : constant := 1024;
   Max_Prover_Len    : constant := 64;

   --  ============================================================
   --  VC Status
   --  ============================================================

   type VC_Status is (
      VC_Unprocessed,
      VC_Valid,
      VC_Invalid,
      VC_Unknown,
      VC_Timeout,
      VC_Error,
      VC_Manual
   );

   --  ============================================================
   --  Complexity Category
   --  ============================================================

   type Complexity_Category is (
      Trivial,       --  <10 steps
      Simple,        --  10-100 steps
      Medium,        --  100-1000 steps
      Complex,       --  1000-10000 steps
      Very_Complex   --  >10000 steps
   );

   --  ============================================================
   --  Bounded Strings
   --  ============================================================

   subtype Formula_Index is Positive range 1 .. Max_Formula_Len;
   subtype Formula_Length is Natural range 0 .. Max_Formula_Len;
   subtype Formula_String is String (Formula_Index);

   subtype Reason_Index is Positive range 1 .. Max_Reason_Len;
   subtype Reason_Length is Natural range 0 .. Max_Reason_Len;
   subtype Reason_String is String (Reason_Index);

   subtype Prover_Index is Positive range 1 .. Max_Prover_Len;
   subtype Prover_Length is Natural range 0 .. Max_Prover_Len;
   subtype Prover_String is String (Prover_Index);

   --  ============================================================
   --  Verification Condition Record
   --  ============================================================

   type VC_Record is record
      ID           : Natural;
      PO_ID        : Natural;  --  Parent proof obligation
      Status       : VC_Status;
      Complexity   : Complexity_Category;
      Formula      : Formula_String;
      Formula_Len  : Formula_Length;
      Reason       : Reason_String;
      Reason_Len   : Reason_Length;
      Step_Count   : Natural;
      Time_MS      : Natural;
      Prover       : Prover_String;
      Prover_Len   : Prover_Length;
      Trivial_Flag : Boolean;  --  Trivially true
   end record;

   --  Empty VC constant
   Empty_VC : constant VC_Record := (
      ID          => 0,
      PO_ID       => 0,
      Status      => VC_Unprocessed,
      Complexity  => Trivial,
      Formula     => (others => ' '),
      Formula_Len => 0,
      Reason      => (others => ' '),
      Reason_Len  => 0,
      Step_Count  => 0,
      Time_MS     => 0,
      Prover      => (others => ' '),
      Prover_Len  => 0,
      Trivial_Flag => False
   );

   --  ============================================================
   --  Manual Proof Record
   --  ============================================================

   Max_Justification_Len : constant := 2048;
   Max_Reviewer_Len      : constant := 128;
   Max_Date_Len          : constant := 10;  --  YYYY-MM-DD

   subtype Just_Index is Positive range 1 .. Max_Justification_Len;
   subtype Just_Length is Natural range 0 .. Max_Justification_Len;
   subtype Just_String is String (Just_Index);

   subtype Reviewer_Index is Positive range 1 .. Max_Reviewer_Len;
   subtype Reviewer_Length is Natural range 0 .. Max_Reviewer_Len;
   subtype Reviewer_String is String (Reviewer_Index);

   subtype Date_String is String (1 .. Max_Date_Len);

   type Manual_Proof_Record is record
      VC_ID         : Natural;
      Justification : Just_String;
      Just_Len      : Just_Length;
      Reviewer      : Reviewer_String;
      Reviewer_Len  : Reviewer_Length;
      Review_Date   : Date_String;
      Approved      : Boolean;
   end record;

   --  Empty manual proof constant
   Empty_Manual_Proof : constant Manual_Proof_Record := (
      VC_ID         => 0,
      Justification => (others => ' '),
      Just_Len      => 0,
      Reviewer      => (others => ' '),
      Reviewer_Len  => 0,
      Review_Date   => "0000-00-00",
      Approved      => False
   );

   --  ============================================================
   --  Predicates
   --  ============================================================

   function Is_Valid_VC (VC : VC_Record) return Boolean is
     (VC.ID > 0);

   function Is_Discharged (VC : VC_Record) return Boolean is
     (VC.Status in VC_Valid | VC_Manual);

   function Is_Failed (VC : VC_Record) return Boolean is
     (VC.Status = VC_Invalid);

   --  ============================================================
   --  Complexity Classification
   --  ============================================================

   function Get_Complexity (Step_Count : Natural) return Complexity_Category is
     (if Step_Count < 10 then Trivial
      elsif Step_Count < 100 then Simple
      elsif Step_Count < 1000 then Medium
      elsif Step_Count < 10000 then Complex
      else Very_Complex);

   --  ============================================================
   --  Operations
   --  ============================================================

   --  Create a new VC record
   procedure Create_VC
     (ID     : Natural;
      PO_ID  : Natural;
      VC     : out VC_Record)
   with
      Pre  => ID > 0,
      Post => Is_Valid_VC (VC) and then VC.ID = ID and then VC.PO_ID = PO_ID;

   --  Update VC status
   procedure Update_VC_Status
     (VC     : in Out VC_Record;
      Status : VC_Status;
      Steps  : Natural;
      Time   : Natural)
   with
      Post => VC.Status = Status and then
              VC.Step_Count = Steps and then
              VC.Time_MS = Time;

   --  Set prover name
   procedure Set_VC_Prover
     (VC   : in Out VC_Record;
      Name : String)
   with
      Pre => Name'Length <= Max_Prover_Len;

   --  Set formula
   procedure Set_Formula
     (VC      : in Out VC_Record;
      Formula : String)
   with
      Pre => Formula'Length <= Max_Formula_Len;

   --  Set reason/description
   procedure Set_Reason
     (VC     : in Out VC_Record;
      Reason : String)
   with
      Pre => Reason'Length <= Max_Reason_Len;

   --  Get status name as string
   function Status_Name (S : VC_Status) return String;

   --  Get complexity name as string
   function Complexity_Name (C : Complexity_Category) return String;

end Verification_Condition;
