--  STUNIR DO-333 Proof Obligation Manager Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;       use Ada.Text_IO;
with Proof_Obligation;  use Proof_Obligation;
with PO_Manager;        use PO_Manager;

procedure Test_PO_Manager is

   Total_Tests  : Natural := 0;
   Passed_Tests : Natural := 0;

   procedure Test (Name : String; Condition : Boolean) is
   begin
      Total_Tests := Total_Tests + 1;
      if Condition then
         Passed_Tests := Passed_Tests + 1;
         Put_Line ("  [PASS] " & Name);
      else
         Put_Line ("  [FAIL] " & Name);
      end if;
   end Test;

begin
   Put_Line ("=== Proof Obligation Manager Tests ===");
   New_Line;

   --  Test 1: Create PO
   declare
      PO : Proof_Obligation_Record;
   begin
      Create_PO (1, PO_Precondition, DAL_A, "test.ads", "Test_Proc", 10, 5, PO);
      Test ("PO ID is 1", PO.ID = 1);
      Test ("PO kind is Precondition", PO.Kind = PO_Precondition);
      Test ("PO criticality is DAL-A", PO.Criticality = DAL_A);
      Test ("PO line is 10", PO.Line = 10);
      Test ("PO is valid", Is_Valid_PO (PO));
      Test ("PO not proved initially", not Is_Proved (PO));
      Test ("PO is critical", Is_Critical (PO));
   end;

   --  Test 2: Update status
   declare
      PO : Proof_Obligation_Record;
   begin
      Create_PO (1, PO_Assert, DAL_B, "test.ads", "Test", 10, 1, PO);
      Test ("PO unproved initially", PO.Status = PO_Unproved);
      Update_Status (PO, PO_Proved, 100, 500);
      Test ("PO proved after update", PO.Status = PO_Proved);
      Test ("PO time updated", PO.Prover_Time = 100);
      Test ("PO steps updated", PO.Step_Count = 500);
      Test ("Is_Proved returns true", Is_Proved (PO));
   end;

   --  Test 3: Empty collection
   declare
      Coll : PO_Collection;
   begin
      Initialize (Coll);
      Test ("Empty collection has count 0", Coll.Count = 0);
      Test ("Is_Empty returns true", Is_Empty (Coll));
   end;

   --  Test 4: Add PO to collection
   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
   begin
      Initialize (Coll);
      Create_PO (1, PO_Precondition, DAL_A, "test.ads", "Proc", 10, 1, PO);
      Add_PO (Coll, PO, Success);
      Test ("Add PO succeeds", Success);
      Test ("Collection count is 1", Coll.Count = 1);
      Test ("Collection not empty", not Is_Empty (Coll));
   end;

   --  Test 5: Find PO
   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
      Idx     : Natural;
   begin
      Initialize (Coll);
      Create_PO (42, PO_Postcondition, DAL_B, "test.ads", "Proc", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      Idx := Find_PO (Coll, 42);
      Test ("Find_PO finds ID 42", Idx > 0);

      Idx := Find_PO (Coll, 999);
      Test ("Find_PO returns 0 for missing", Idx = 0);
   end;

   --  Test 6: Coverage metrics
   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
      Metrics : Coverage_Metrics;
   begin
      Initialize (Coll);

      --  Add proved PO
      Create_PO (1, PO_Assert, DAL_C, "test.ads", "P1", 10, 1, PO);
      Update_Status (PO, PO_Proved, 50, 100);
      Add_PO (Coll, PO, Success);

      --  Add unproved PO
      Create_PO (2, PO_Range_Check, DAL_D, "test.ads", "P2", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      Metrics := Get_Metrics (Coll);
      Test ("Total POs is 2", Metrics.Total_POs = 2);
      Test ("Proved POs is 1", Metrics.Proved_POs = 1);
      Test ("Unproved POs is 1", Metrics.Unproved_POs = 1);
      Test ("Coverage is 50%", Metrics.Coverage_Pct = 50);
   end;

   --  Test 7: Unproven analysis
   declare
      Coll     : PO_Collection;
      PO       : Proof_Obligation_Record;
      Success  : Boolean;
      Analysis : Unproven_Analysis;
   begin
      Initialize (Coll);

      --  Add DAL-A unproved
      Create_PO (1, PO_Precondition, DAL_A, "test.ads", "Critical", 10, 1, PO);
      Add_PO (Coll, PO, Success);

      --  Add DAL-C unproved
      Create_PO (2, PO_Assert, DAL_C, "test.ads", "Safety", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      Analysis := Get_Unproven_Analysis (Coll);
      Test ("Critical unproven is 1", Analysis.Critical_Unproven = 1);
      Test ("Safety unproven is 2", Analysis.Safety_Unproven = 2);
   end;

   --  Test 8: Count functions
   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
   begin
      Initialize (Coll);

      Create_PO (1, PO_Precondition, DAL_A, "t.ads", "P", 10, 1, PO);
      Update_Status (PO, PO_Proved, 50, 100);
      Add_PO (Coll, PO, Success);

      Create_PO (2, PO_Precondition, DAL_B, "t.ads", "P", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      Test ("Count by kind Precondition is 2",
            Count_By_Kind (Coll, PO_Precondition) = 2);
      Test ("Count by status Proved is 1",
            Count_By_Status (Coll, PO_Proved) = 1);
      Test ("Count by criticality DAL-A is 1",
            Count_By_Criticality (Coll, DAL_A) = 1);
   end;

   --  Test 9: All critical proved
   declare
      Coll    : PO_Collection;
      PO      : Proof_Obligation_Record;
      Success : Boolean;
   begin
      Initialize (Coll);

      --  Add proved critical PO
      Create_PO (1, PO_Assert, DAL_A, "t.ads", "P", 10, 1, PO);
      Update_Status (PO, PO_Proved, 50, 100);
      Add_PO (Coll, PO, Success);

      Test ("All critical proved (only proved)", All_Critical_Proved (Coll));

      --  Add unproved critical PO
      Create_PO (2, PO_Range_Check, DAL_B, "t.ads", "P", 20, 1, PO);
      Add_PO (Coll, PO, Success);

      Test ("Not all critical proved", not All_Critical_Proved (Coll));
   end;

   --  Test 10: Status and kind names
   Test ("Status name for Proved",
         Status_Name (PO_Proved) = "Proved");
   Test ("Status name for Unproved",
         Status_Name (PO_Unproved) = "Unproved");
   Test ("Kind name for Precondition",
         Kind_Name (PO_Precondition) = "Precondition");
   Test ("Criticality name for DAL-A",
         Criticality_Name (DAL_A) = "DAL-A (Catastrophic)");

   --  Summary
   New_Line;
   Put_Line ("=== Test Summary ===");
   Put_Line ("Passed:" & Natural'Image (Passed_Tests) &
             " /" & Natural'Image (Total_Tests));

   if Passed_Tests = Total_Tests then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED!");
   end if;

end Test_PO_Manager;
