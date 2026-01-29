--  STUNIR DO-333 Verification Condition Tracker
--  Manages collections of verification conditions
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides VC collection management:
--  - Add/update verification conditions
--  - Coverage report generation
--  - Complexity analysis
--  - Manual proof tracking
--
--  DO-333 Objective: FM.4 (Verification Condition Management)

pragma SPARK_Mode (On);

with Verification_Condition; use Verification_Condition;

package VC_Tracker is

   --  ============================================================
   --  VC Collection
   --  ============================================================

   subtype VC_Index is Positive range 1 .. Max_VC_Count;
   subtype VC_Count is Natural range 0 .. Max_VC_Count;

   type VC_Array is array (VC_Index) of VC_Record;

   type VC_Collection is record
      Items : VC_Array;
      Count : VC_Count;
   end record;

   --  Empty collection constant
   Empty_VC_Collection : constant VC_Collection := (
      Items => (others => Empty_VC),
      Count => 0
   );

   --  ============================================================
   --  Manual Proof Collection
   --  ============================================================

   Max_Manual_Proofs : constant := 500;

   subtype Manual_Index is Positive range 1 .. Max_Manual_Proofs;
   subtype Manual_Count is Natural range 0 .. Max_Manual_Proofs;

   type Manual_Array is array (Manual_Index) of Manual_Proof_Record;

   type Manual_Proof_Collection is record
      Items : Manual_Array;
      Count : Manual_Count;
   end record;

   Empty_Manual_Collection : constant Manual_Proof_Collection := (
      Items => (others => Empty_Manual_Proof),
      Count => 0
   );

   --  ============================================================
   --  VC Coverage Report
   --  ============================================================

   type Complexity_Count_Array is array (Complexity_Category) of Natural;

   type VC_Coverage_Report is record
      Total_VCs     : Natural;
      Valid_VCs     : Natural;
      Invalid_VCs   : Natural;
      Unknown_VCs   : Natural;
      Timeout_VCs   : Natural;
      Manual_VCs    : Natural;
      Error_VCs     : Natural;
      By_Complexity : Complexity_Count_Array;
      Coverage_Pct  : Natural;  --  0-100
   end record;

   Empty_Coverage_Report : constant VC_Coverage_Report := (
      Total_VCs     => 0,
      Valid_VCs     => 0,
      Invalid_VCs   => 0,
      Unknown_VCs   => 0,
      Timeout_VCs   => 0,
      Manual_VCs    => 0,
      Error_VCs     => 0,
      By_Complexity => (others => 0),
      Coverage_Pct  => 0
   );

   --  ============================================================
   --  Complexity Analysis
   --  ============================================================

   type Complexity_Analysis is record
      Avg_Steps      : Natural;
      Max_Steps      : Natural;
      Min_Steps      : Natural;
      Avg_Time_MS    : Natural;
      Max_Time_MS    : Natural;
      Complex_Count  : Natural;  --  Complex + Very_Complex
      Trivial_Count  : Natural;
   end record;

   Empty_Complexity_Analysis : constant Complexity_Analysis := (
      Avg_Steps     => 0,
      Max_Steps     => 0,
      Min_Steps     => Natural'Last,
      Avg_Time_MS   => 0,
      Max_Time_MS   => 0,
      Complex_Count => 0,
      Trivial_Count => 0
   );

   --  ============================================================
   --  Collection Operations
   --  ============================================================

   --  Initialize collection
   procedure Initialize (Coll : out VC_Collection)
   with
      Post => Coll.Count = 0;

   --  Add a VC to collection
   procedure Add_VC
     (Coll    : in Out VC_Collection;
      VC      : VC_Record;
      Success : out Boolean)
   with
      Pre  => Is_Valid_VC (VC),
      Post => (if Success then Coll.Count = Coll.Count'Old + 1);

   --  Update VC status by ID
   procedure Update_Status
     (Coll   : in Out VC_Collection;
      VC_ID  : Natural;
      Status : VC_Status;
      Steps  : Natural;
      Time   : Natural);

   --  Find VC by ID
   function Find_VC
     (Coll  : VC_Collection;
      VC_ID : Natural) return Natural;  --  Returns index, 0 if not found

   --  Get VC by index
   function Get_VC
     (Coll  : VC_Collection;
      Index : VC_Index) return VC_Record
   with
      Pre => Index <= Coll.Count;

   --  Get VCs for a specific PO
   function Count_VCs_For_PO
     (Coll  : VC_Collection;
      PO_ID : Natural) return Natural;

   --  ============================================================
   --  Manual Proof Operations
   --  ============================================================

   --  Initialize manual proof collection
   procedure Initialize_Manual (Coll : out Manual_Proof_Collection)
   with
      Post => Coll.Count = 0;

   --  Add manual proof
   procedure Add_Manual_Proof
     (Manual_Coll : in Out Manual_Proof_Collection;
      Proof       : Manual_Proof_Record;
      Success     : out Boolean)
   with
      Post => (if Success then Manual_Coll.Count = Manual_Coll.Count'Old + 1);

   --  Find manual proof for VC
   function Find_Manual_Proof
     (Manual_Coll : Manual_Proof_Collection;
      VC_ID       : Natural) return Natural;

   --  ============================================================
   --  Analysis and Reporting
   --  ============================================================

   --  Calculate coverage report
   function Get_Coverage (Coll : VC_Collection) return VC_Coverage_Report;

   --  Analyze complexity
   function Get_Complexity_Analysis
     (Coll : VC_Collection) return Complexity_Analysis;

   --  Count VCs by status
   function Count_By_Status
     (Coll   : VC_Collection;
      Status : VC_Status) return Natural;

   --  Count VCs by complexity
   function Count_By_Complexity
     (Coll       : VC_Collection;
      Complexity : Complexity_Category) return Natural;

   --  ============================================================
   --  Queries
   --  ============================================================

   --  Check if all VCs are discharged
   function All_Discharged
     (Coll : VC_Collection) return Boolean;

   --  Check if collection is empty
   function Is_Empty (Coll : VC_Collection) return Boolean is
     (Coll.Count = 0);

   --  Get total coverage percentage
   function Coverage_Percentage
     (Coll : VC_Collection) return Natural;

   --  Get total step count
   function Total_Steps
     (Coll : VC_Collection) return Natural;

   --  Get total time
   function Total_Time_MS
     (Coll : VC_Collection) return Natural;

end VC_Tracker;
