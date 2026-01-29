--  STUNIR DO-333 Proof Obligation Manager
--  Manages collections of proof obligations
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides PO collection management:
--  - Add/update proof obligations
--  - Coverage metrics calculation
--  - Unproven analysis
--  - Prioritization by criticality
--
--  DO-333 Objectives: FM.2, FM.3 (Proofs, Coverage)

pragma SPARK_Mode (On);

with Proof_Obligation; use Proof_Obligation;

package PO_Manager is

   --  ============================================================
   --  PO Collection
   --  ============================================================

   subtype PO_Index is Positive range 1 .. Max_PO_Count;
   subtype PO_Count is Natural range 0 .. Max_PO_Count;

   type PO_Array is array (PO_Index) of Proof_Obligation_Record;

   type PO_Collection is record
      Items : PO_Array;
      Count : PO_Count;
   end record;

   --  Empty collection constant
   Empty_Collection : constant PO_Collection := (
      Items => (others => Empty_PO),
      Count => 0
   );

   --  ============================================================
   --  Coverage Metrics
   --  ============================================================

   type Coverage_Metrics is record
      Total_POs     : Natural;
      Proved_POs    : Natural;
      Unproved_POs  : Natural;
      Timeout_POs   : Natural;
      Error_POs     : Natural;
      Justified_POs : Natural;
      Flow_POs      : Natural;
      Coverage_Pct  : Natural;  --  0-100
   end record;

   Empty_Metrics : constant Coverage_Metrics := (
      Total_POs     => 0,
      Proved_POs    => 0,
      Unproved_POs  => 0,
      Timeout_POs   => 0,
      Error_POs     => 0,
      Justified_POs => 0,
      Flow_POs      => 0,
      Coverage_Pct  => 0
   );

   --  ============================================================
   --  Unproven Analysis
   --  ============================================================

   type Kind_Count_Array is array (PO_Kind) of Natural;
   type DAL_Count_Array is array (Criticality_Level) of Natural;

   type Unproven_Analysis is record
      By_Kind           : Kind_Count_Array;
      By_DAL            : DAL_Count_Array;
      Critical_Unproven : Natural;  --  DAL-A + DAL-B unproven
      Safety_Unproven   : Natural;  --  DAL-A + DAL-B + DAL-C unproven
   end record;

   Empty_Analysis : constant Unproven_Analysis := (
      By_Kind           => (others => 0),
      By_DAL            => (others => 0),
      Critical_Unproven => 0,
      Safety_Unproven   => 0
   );

   --  ============================================================
   --  Collection Operations
   --  ============================================================

   --  Initialize collection
   procedure Initialize (Coll : out PO_Collection)
   with
      Post => Coll.Count = 0;

   --  Add a PO to collection
   procedure Add_PO
     (Coll    : in Out PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : out Boolean)
   with
      Pre  => Is_Valid_PO (PO),
      Post => (if Success then Coll.Count = Coll.Count'Old + 1);

   --  Update PO status by ID
   procedure Update_Status
     (Coll   : in Out PO_Collection;
      PO_ID  : Natural;
      Status : PO_Status;
      Time   : Natural;
      Steps  : Natural);

   --  Find PO by ID
   function Find_PO
     (Coll  : PO_Collection;
      PO_ID : Natural) return Natural;  --  Returns index, 0 if not found

   --  Get PO by index
   function Get_PO
     (Coll  : PO_Collection;
      Index : PO_Index) return Proof_Obligation_Record
   with
      Pre => Index <= Coll.Count;

   --  ============================================================
   --  Metrics and Analysis
   --  ============================================================

   --  Calculate coverage metrics
   function Get_Metrics (Coll : PO_Collection) return Coverage_Metrics;

   --  Analyze unproven POs
   function Get_Unproven_Analysis
     (Coll : PO_Collection) return Unproven_Analysis;

   --  Count POs by status
   function Count_By_Status
     (Coll   : PO_Collection;
      Status : PO_Status) return Natural;

   --  Count POs by kind
   function Count_By_Kind
     (Coll : PO_Collection;
      Kind : PO_Kind) return Natural;

   --  Count POs by criticality
   function Count_By_Criticality
     (Coll        : PO_Collection;
      Criticality : Criticality_Level) return Natural;

   --  ============================================================
   --  Prioritization
   --  ============================================================

   --  Sort by criticality (DAL-A first)
   procedure Prioritize_By_Criticality
     (Coll : in Out PO_Collection);

   --  Sort by status (unproved first)
   procedure Prioritize_By_Status
     (Coll : in Out PO_Collection);

   --  ============================================================
   --  Queries
   --  ============================================================

   --  Check if all critical POs are proved
   function All_Critical_Proved
     (Coll : PO_Collection) return Boolean;

   --  Check if all safety-related POs are proved
   function All_Safety_Proved
     (Coll : PO_Collection) return Boolean;

   --  Check if collection is empty
   function Is_Empty (Coll : PO_Collection) return Boolean is
     (Coll.Count = 0);

   --  Get total coverage percentage
   function Coverage_Percentage
     (Coll : PO_Collection) return Natural;

end PO_Manager;
