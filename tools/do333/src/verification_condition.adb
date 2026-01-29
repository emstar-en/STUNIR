--  STUNIR DO-333 Verification Condition Types
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Verification_Condition is

   --  ============================================================
   --  Create VC
   --  ============================================================

   procedure Create_VC
     (ID     : Natural;
      PO_ID  : Natural;
      VC     : out VC_Record)
   is
   begin
      VC := Empty_VC;
      VC.ID := ID;
      VC.PO_ID := PO_ID;
      VC.Status := VC_Unprocessed;
      VC.Complexity := Trivial;
   end Create_VC;

   --  ============================================================
   --  Update VC Status
   --  ============================================================

   procedure Update_VC_Status
     (VC     : in Out VC_Record;
      Status : VC_Status;
      Steps  : Natural;
      Time   : Natural)
   is
   begin
      VC.Status := Status;
      VC.Step_Count := Steps;
      VC.Time_MS := Time;
      VC.Complexity := Get_Complexity (Steps);
   end Update_VC_Status;

   --  ============================================================
   --  Set VC Prover
   --  ============================================================

   procedure Set_VC_Prover
     (VC   : in Out VC_Record;
      Name : String)
   is
      Buf : Prover_String := (others => ' ');
   begin
      for I in Name'Range loop
         Buf (I - Name'First + 1) := Name (I);
      end loop;
      VC.Prover := Buf;
      VC.Prover_Len := Name'Length;
   end Set_VC_Prover;

   --  ============================================================
   --  Set Formula
   --  ============================================================

   procedure Set_Formula
     (VC      : in Out VC_Record;
      Formula : String)
   is
      Buf : Formula_String := (others => ' ');
   begin
      for I in Formula'Range loop
         Buf (I - Formula'First + 1) := Formula (I);
      end loop;
      VC.Formula := Buf;
      VC.Formula_Len := Formula'Length;
   end Set_Formula;

   --  ============================================================
   --  Set Reason
   --  ============================================================

   procedure Set_Reason
     (VC     : in Out VC_Record;
      Reason : String)
   is
      Buf : Reason_String := (others => ' ');
   begin
      for I in Reason'Range loop
         Buf (I - Reason'First + 1) := Reason (I);
      end loop;
      VC.Reason := Buf;
      VC.Reason_Len := Reason'Length;
   end Set_Reason;

   --  ============================================================
   --  Status Name
   --  ============================================================

   function Status_Name (S : VC_Status) return String is
   begin
      case S is
         when VC_Unprocessed => return "Unprocessed";
         when VC_Valid       => return "Valid";
         when VC_Invalid     => return "Invalid";
         when VC_Unknown     => return "Unknown";
         when VC_Timeout     => return "Timeout";
         when VC_Error       => return "Error";
         when VC_Manual      => return "Manual";
      end case;
   end Status_Name;

   --  ============================================================
   --  Complexity Name
   --  ============================================================

   function Complexity_Name (C : Complexity_Category) return String is
   begin
      case C is
         when Trivial      => return "Trivial (<10 steps)";
         when Simple       => return "Simple (10-100 steps)";
         when Medium       => return "Medium (100-1000 steps)";
         when Complex      => return "Complex (1000-10000 steps)";
         when Very_Complex => return "Very Complex (>10000 steps)";
      end case;
   end Complexity_Name;

end Verification_Condition;
