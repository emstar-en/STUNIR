--  STUNIR DO-333 Verification Condition Tracker
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body VC_Tracker is

   --  ============================================================
   --  Initialize
   --  ============================================================

   procedure Initialize (Coll : out VC_Collection) is
   begin
      Coll := Empty_VC_Collection;
   end Initialize;

   --  ============================================================
   --  Add VC
   --  ============================================================

   procedure Add_VC
     (Coll    : in Out VC_Collection;
      VC      : VC_Record;
      Success : out Boolean)
   is
   begin
      if Coll.Count >= Max_VC_Count then
         Success := False;
         return;
      end if;

      Coll.Count := Coll.Count + 1;
      Coll.Items (Coll.Count) := VC;
      Success := True;
   end Add_VC;

   --  ============================================================
   --  Update Status
   --  ============================================================

   procedure Update_Status
     (Coll   : in Out VC_Collection;
      VC_ID  : Natural;
      Status : VC_Status;
      Steps  : Natural;
      Time   : Natural)
   is
      Index : constant Natural := Find_VC (Coll, VC_ID);
   begin
      if Index > 0 and then Index <= Coll.Count then
         Update_VC_Status (Coll.Items (Index), Status, Steps, Time);
      end if;
   end Update_Status;

   --  ============================================================
   --  Find VC
   --  ============================================================

   function Find_VC
     (Coll  : VC_Collection;
      VC_ID : Natural) return Natural
   is
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).ID = VC_ID then
            return I;
         end if;
      end loop;
      return 0;
   end Find_VC;

   --  ============================================================
   --  Get VC
   --  ============================================================

   function Get_VC
     (Coll  : VC_Collection;
      Index : VC_Index) return VC_Record
   is
   begin
      return Coll.Items (Index);
   end Get_VC;

   --  ============================================================
   --  Count VCs For PO
   --  ============================================================

   function Count_VCs_For_PO
     (Coll  : VC_Collection;
      PO_ID : Natural) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).PO_ID = PO_ID then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_VCs_For_PO;

   --  ============================================================
   --  Initialize Manual
   --  ============================================================

   procedure Initialize_Manual (Coll : out Manual_Proof_Collection) is
   begin
      Coll := Empty_Manual_Collection;
   end Initialize_Manual;

   --  ============================================================
   --  Add Manual Proof
   --  ============================================================

   procedure Add_Manual_Proof
     (Manual_Coll : in Out Manual_Proof_Collection;
      Proof       : Manual_Proof_Record;
      Success     : out Boolean)
   is
   begin
      if Manual_Coll.Count >= Max_Manual_Proofs then
         Success := False;
         return;
      end if;

      Manual_Coll.Count := Manual_Coll.Count + 1;
      Manual_Coll.Items (Manual_Coll.Count) := Proof;
      Success := True;
   end Add_Manual_Proof;

   --  ============================================================
   --  Find Manual Proof
   --  ============================================================

   function Find_Manual_Proof
     (Manual_Coll : Manual_Proof_Collection;
      VC_ID       : Natural) return Natural
   is
   begin
      for I in 1 .. Manual_Coll.Count loop
         if Manual_Coll.Items (I).VC_ID = VC_ID then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Manual_Proof;

   --  ============================================================
   --  Get Coverage
   --  ============================================================

   function Get_Coverage (Coll : VC_Collection) return VC_Coverage_Report is
      R : VC_Coverage_Report := Empty_Coverage_Report;
   begin
      R.Total_VCs := Coll.Count;

      for I in 1 .. Coll.Count loop
         --  Count by status
         case Coll.Items (I).Status is
            when VC_Valid       => R.Valid_VCs := R.Valid_VCs + 1;
            when VC_Invalid     => R.Invalid_VCs := R.Invalid_VCs + 1;
            when VC_Unknown     => R.Unknown_VCs := R.Unknown_VCs + 1;
            when VC_Timeout     => R.Timeout_VCs := R.Timeout_VCs + 1;
            when VC_Manual      => R.Manual_VCs := R.Manual_VCs + 1;
            when VC_Error       => R.Error_VCs := R.Error_VCs + 1;
            when VC_Unprocessed => null;
         end case;

         --  Count by complexity
         R.By_Complexity (Coll.Items (I).Complexity) :=
           R.By_Complexity (Coll.Items (I).Complexity) + 1;
      end loop;

      --  Calculate coverage percentage
      if R.Total_VCs > 0 then
         declare
            Discharged : constant Natural := R.Valid_VCs + R.Manual_VCs;
         begin
            R.Coverage_Pct := (Discharged * 100) / R.Total_VCs;
         end;
      end if;

      return R;
   end Get_Coverage;

   --  ============================================================
   --  Get Complexity Analysis
   --  ============================================================

   function Get_Complexity_Analysis
     (Coll : VC_Collection) return Complexity_Analysis
   is
      A          : Complexity_Analysis := Empty_Complexity_Analysis;
      Total_Steps : Natural := 0;
      Total_Time  : Natural := 0;
   begin
      if Coll.Count = 0 then
         A.Min_Steps := 0;
         return A;
      end if;

      for I in 1 .. Coll.Count loop
         --  Update step statistics
         Total_Steps := Total_Steps + Coll.Items (I).Step_Count;
         if Coll.Items (I).Step_Count > A.Max_Steps then
            A.Max_Steps := Coll.Items (I).Step_Count;
         end if;
         if Coll.Items (I).Step_Count < A.Min_Steps then
            A.Min_Steps := Coll.Items (I).Step_Count;
         end if;

         --  Update time statistics
         Total_Time := Total_Time + Coll.Items (I).Time_MS;
         if Coll.Items (I).Time_MS > A.Max_Time_MS then
            A.Max_Time_MS := Coll.Items (I).Time_MS;
         end if;

         --  Count by complexity category
         case Coll.Items (I).Complexity is
            when Trivial =>
               A.Trivial_Count := A.Trivial_Count + 1;
            when Simple | Medium =>
               null;
            when Complex | Very_Complex =>
               A.Complex_Count := A.Complex_Count + 1;
         end case;
      end loop;

      --  Calculate averages
      A.Avg_Steps := Total_Steps / Coll.Count;
      A.Avg_Time_MS := Total_Time / Coll.Count;

      return A;
   end Get_Complexity_Analysis;

   --  ============================================================
   --  Count By Status
   --  ============================================================

   function Count_By_Status
     (Coll   : VC_Collection;
      Status : VC_Status) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).Status = Status then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_By_Status;

   --  ============================================================
   --  Count By Complexity
   --  ============================================================

   function Count_By_Complexity
     (Coll       : VC_Collection;
      Complexity : Complexity_Category) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).Complexity = Complexity then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_By_Complexity;

   --  ============================================================
   --  All Discharged
   --  ============================================================

   function All_Discharged
     (Coll : VC_Collection) return Boolean
   is
   begin
      for I in 1 .. Coll.Count loop
         if not Is_Discharged (Coll.Items (I)) then
            return False;
         end if;
      end loop;
      return True;
   end All_Discharged;

   --  ============================================================
   --  Coverage Percentage
   --  ============================================================

   function Coverage_Percentage
     (Coll : VC_Collection) return Natural
   is
      Report : constant VC_Coverage_Report := Get_Coverage (Coll);
   begin
      return Report.Coverage_Pct;
   end Coverage_Percentage;

   --  ============================================================
   --  Total Steps
   --  ============================================================

   function Total_Steps
     (Coll : VC_Collection) return Natural
   is
      Sum : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         Sum := Sum + Coll.Items (I).Step_Count;
      end loop;
      return Sum;
   end Total_Steps;

   --  ============================================================
   --  Total Time MS
   --  ============================================================

   function Total_Time_MS
     (Coll : VC_Collection) return Natural
   is
      Sum : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         Sum := Sum + Coll.Items (I).Time_MS;
      end loop;
      return Sum;
   end Total_Time_MS;

end VC_Tracker;
