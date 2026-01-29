--  STUNIR DO-333 Proof Obligation Manager
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body PO_Manager is

   --  ============================================================
   --  Initialize
   --  ============================================================

   procedure Initialize (Coll : out PO_Collection) is
   begin
      Coll := Empty_Collection;
   end Initialize;

   --  ============================================================
   --  Add PO
   --  ============================================================

   procedure Add_PO
     (Coll    : in Out PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : out Boolean)
   is
   begin
      if Coll.Count >= Max_PO_Count then
         Success := False;
         return;
      end if;

      Coll.Count := Coll.Count + 1;
      Coll.Items (Coll.Count) := PO;
      Success := True;
   end Add_PO;

   --  ============================================================
   --  Update Status
   --  ============================================================

   procedure Update_Status
     (Coll   : in Out PO_Collection;
      PO_ID  : Natural;
      Status : PO_Status;
      Time   : Natural;
      Steps  : Natural)
   is
      Index : constant Natural := Find_PO (Coll, PO_ID);
   begin
      if Index > 0 and then Index <= Coll.Count then
         Proof_Obligation.Update_Status
           (Coll.Items (Index), Status, Time, Steps);
      end if;
   end Update_Status;

   --  ============================================================
   --  Find PO
   --  ============================================================

   function Find_PO
     (Coll  : PO_Collection;
      PO_ID : Natural) return Natural
   is
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).ID = PO_ID then
            return I;
         end if;
      end loop;
      return 0;
   end Find_PO;

   --  ============================================================
   --  Get PO
   --  ============================================================

   function Get_PO
     (Coll  : PO_Collection;
      Index : PO_Index) return Proof_Obligation_Record
   is
   begin
      return Coll.Items (Index);
   end Get_PO;

   --  ============================================================
   --  Get Metrics
   --  ============================================================

   function Get_Metrics (Coll : PO_Collection) return Coverage_Metrics is
      M : Coverage_Metrics := Empty_Metrics;
   begin
      M.Total_POs := Coll.Count;

      for I in 1 .. Coll.Count loop
         case Coll.Items (I).Status is
            when PO_Proved =>
               M.Proved_POs := M.Proved_POs + 1;
            when PO_Unproved =>
               M.Unproved_POs := M.Unproved_POs + 1;
            when PO_Timeout =>
               M.Timeout_POs := M.Timeout_POs + 1;
            when PO_Error =>
               M.Error_POs := M.Error_POs + 1;
            when PO_Manually_Justified =>
               M.Justified_POs := M.Justified_POs + 1;
            when PO_Flow_Proved =>
               M.Flow_POs := M.Flow_POs + 1;
            when PO_Skipped =>
               null;  --  Not counted
         end case;
      end loop;

      --  Calculate coverage percentage
      if M.Total_POs > 0 then
         declare
            Proved_Total : constant Natural :=
              M.Proved_POs + M.Flow_POs + M.Justified_POs;
         begin
            M.Coverage_Pct := (Proved_Total * 100) / M.Total_POs;
         end;
      end if;

      return M;
   end Get_Metrics;

   --  ============================================================
   --  Get Unproven Analysis
   --  ============================================================

   function Get_Unproven_Analysis
     (Coll : PO_Collection) return Unproven_Analysis
   is
      A : Unproven_Analysis := Empty_Analysis;
   begin
      for I in 1 .. Coll.Count loop
         if not Is_Proved (Coll.Items (I)) then
            --  Count by kind
            A.By_Kind (Coll.Items (I).Kind) :=
              A.By_Kind (Coll.Items (I).Kind) + 1;

            --  Count by DAL
            A.By_DAL (Coll.Items (I).Criticality) :=
              A.By_DAL (Coll.Items (I).Criticality) + 1;

            --  Track critical and safety
            case Coll.Items (I).Criticality is
               when DAL_A | DAL_B =>
                  A.Critical_Unproven := A.Critical_Unproven + 1;
                  A.Safety_Unproven := A.Safety_Unproven + 1;
               when DAL_C =>
                  A.Safety_Unproven := A.Safety_Unproven + 1;
               when DAL_D | DAL_E =>
                  null;
            end case;
         end if;
      end loop;

      return A;
   end Get_Unproven_Analysis;

   --  ============================================================
   --  Count By Status
   --  ============================================================

   function Count_By_Status
     (Coll   : PO_Collection;
      Status : PO_Status) return Natural
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
   --  Count By Kind
   --  ============================================================

   function Count_By_Kind
     (Coll : PO_Collection;
      Kind : PO_Kind) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).Kind = Kind then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_By_Kind;

   --  ============================================================
   --  Count By Criticality
   --  ============================================================

   function Count_By_Criticality
     (Coll        : PO_Collection;
      Criticality : Criticality_Level) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Coll.Count loop
         if Coll.Items (I).Criticality = Criticality then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_By_Criticality;

   --  ============================================================
   --  Prioritize By Criticality
   --  ============================================================

   procedure Prioritize_By_Criticality
     (Coll : in Out PO_Collection)
   is
      --  Simple bubble sort (stable for small collections)
      Swapped : Boolean;
      Temp    : Proof_Obligation_Record;
   begin
      if Coll.Count <= 1 then
         return;
      end if;

      loop
         Swapped := False;
         for I in 1 .. Coll.Count - 1 loop
            if Criticality_Level'Pos (Coll.Items (I).Criticality) >
               Criticality_Level'Pos (Coll.Items (I + 1).Criticality)
            then
               Temp := Coll.Items (I);
               Coll.Items (I) := Coll.Items (I + 1);
               Coll.Items (I + 1) := Temp;
               Swapped := True;
            end if;
         end loop;
         exit when not Swapped;
      end loop;
   end Prioritize_By_Criticality;

   --  ============================================================
   --  Prioritize By Status
   --  ============================================================

   procedure Prioritize_By_Status
     (Coll : in Out PO_Collection)
   is
      --  Simple bubble sort (unproved first)
      Swapped : Boolean;
      Temp    : Proof_Obligation_Record;
   begin
      if Coll.Count <= 1 then
         return;
      end if;

      loop
         Swapped := False;
         for I in 1 .. Coll.Count - 1 loop
            --  Unproved should come before proved
            if Is_Proved (Coll.Items (I)) and then
               not Is_Proved (Coll.Items (I + 1))
            then
               Temp := Coll.Items (I);
               Coll.Items (I) := Coll.Items (I + 1);
               Coll.Items (I + 1) := Temp;
               Swapped := True;
            end if;
         end loop;
         exit when not Swapped;
      end loop;
   end Prioritize_By_Status;

   --  ============================================================
   --  All Critical Proved
   --  ============================================================

   function All_Critical_Proved
     (Coll : PO_Collection) return Boolean
   is
   begin
      for I in 1 .. Coll.Count loop
         if Is_Critical (Coll.Items (I)) and then
            not Is_Proved (Coll.Items (I))
         then
            return False;
         end if;
      end loop;
      return True;
   end All_Critical_Proved;

   --  ============================================================
   --  All Safety Proved
   --  ============================================================

   function All_Safety_Proved
     (Coll : PO_Collection) return Boolean
   is
   begin
      for I in 1 .. Coll.Count loop
         if Is_Safety_Related (Coll.Items (I)) and then
            not Is_Proved (Coll.Items (I))
         then
            return False;
         end if;
      end loop;
      return True;
   end All_Safety_Proved;

   --  ============================================================
   --  Coverage Percentage
   --  ============================================================

   function Coverage_Percentage
     (Coll : PO_Collection) return Natural
   is
      Metrics : constant Coverage_Metrics := Get_Metrics (Coll);
   begin
      return Metrics.Coverage_Pct;
   end Coverage_Percentage;

end PO_Manager;
