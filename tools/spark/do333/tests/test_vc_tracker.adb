--  STUNIR DO-333 Verification Condition Tracker Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;            use Ada.Text_IO;
with Verification_Condition; use Verification_Condition;
with VC_Tracker;             use VC_Tracker;

procedure Test_VC_Tracker is

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
   Put_Line ("=== Verification Condition Tracker Tests ===");
   New_Line;

   --  Test 1: Create VC
   declare
      VC : VC_Record;
   begin
      Create_VC (1, 100, VC);
      Test ("VC ID is 1", VC.ID = 1);
      Test ("VC PO_ID is 100", VC.PO_ID = 100);
      Test ("VC status is Unprocessed", VC.Status = VC_Unprocessed);
      Test ("VC is valid", Is_Valid_VC (VC));
      Test ("VC not discharged initially", not Is_Discharged (VC));
   end;

   --  Test 2: Update VC status
   declare
      VC : VC_Record;
   begin
      Create_VC (1, 1, VC);
      Update_VC_Status (VC, VC_Valid, 500, 1000);
      Test ("VC status is Valid", VC.Status = VC_Valid);
      Test ("VC steps is 500", VC.Step_Count = 500);
      Test ("VC time is 1000", VC.Time_MS = 1000);
      Test ("VC is discharged", Is_Discharged (VC));
   end;

   --  Test 3: Complexity classification
   Test ("Trivial < 10 steps", Get_Complexity (5) = Trivial);
   Test ("Simple 10-100 steps", Get_Complexity (50) = Simple);
   Test ("Medium 100-1000 steps", Get_Complexity (500) = Medium);
   Test ("Complex 1000-10000 steps", Get_Complexity (5000) = Complex);
   Test ("Very_Complex > 10000 steps", Get_Complexity (15000) = Very_Complex);

   --  Test 4: Empty collection
   declare
      Coll : VC_Collection;
   begin
      Initialize (Coll);
      Test ("Empty VC collection has count 0", Coll.Count = 0);
      Test ("Is_Empty returns true", Is_Empty (Coll));
   end;

   --  Test 5: Add VC to collection
   declare
      Coll    : VC_Collection;
      VC      : VC_Record;
      Success : Boolean;
   begin
      Initialize (Coll);
      Create_VC (1, 1, VC);
      Add_VC (Coll, VC, Success);
      Test ("Add VC succeeds", Success);
      Test ("VC collection count is 1", Coll.Count = 1);
   end;

   --  Test 6: Find VC
   declare
      Coll    : VC_Collection;
      VC      : VC_Record;
      Success : Boolean;
      Idx     : Natural;
   begin
      Initialize (Coll);
      Create_VC (42, 10, VC);
      Add_VC (Coll, VC, Success);

      Idx := Find_VC (Coll, 42);
      Test ("Find_VC finds ID 42", Idx > 0);

      Idx := Find_VC (Coll, 999);
      Test ("Find_VC returns 0 for missing", Idx = 0);
   end;

   --  Test 7: Coverage report
   declare
      Coll     : VC_Collection;
      VC       : VC_Record;
      Success  : Boolean;
      Coverage : VC_Coverage_Report;
   begin
      Initialize (Coll);

      --  Add valid VC
      Create_VC (1, 1, VC);
      Update_VC_Status (VC, VC_Valid, 100, 200);
      Add_VC (Coll, VC, Success);

      --  Add unknown VC
      Create_VC (2, 1, VC);
      Update_VC_Status (VC, VC_Unknown, 50, 100);
      Add_VC (Coll, VC, Success);

      Coverage := Get_Coverage (Coll);
      Test ("Total VCs is 2", Coverage.Total_VCs = 2);
      Test ("Valid VCs is 1", Coverage.Valid_VCs = 1);
      Test ("Unknown VCs is 1", Coverage.Unknown_VCs = 1);
      Test ("Coverage is 50%", Coverage.Coverage_Pct = 50);
   end;

   --  Test 8: Complexity analysis
   declare
      Coll     : VC_Collection;
      VC       : VC_Record;
      Success  : Boolean;
      Analysis : Complexity_Analysis;
   begin
      Initialize (Coll);

      --  Add trivial VC
      Create_VC (1, 1, VC);
      Update_VC_Status (VC, VC_Valid, 5, 10);
      Add_VC (Coll, VC, Success);

      --  Add complex VC
      Create_VC (2, 1, VC);
      Update_VC_Status (VC, VC_Valid, 5000, 10000);
      Add_VC (Coll, VC, Success);

      Analysis := Get_Complexity_Analysis (Coll);
      Test ("Max steps is 5000", Analysis.Max_Steps = 5000);
      Test ("Min steps is 5", Analysis.Min_Steps = 5);
      Test ("Trivial count is 1", Analysis.Trivial_Count = 1);
      Test ("Complex count is 1", Analysis.Complex_Count = 1);
   end;

   --  Test 9: Count VCs for PO
   declare
      Coll    : VC_Collection;
      VC      : VC_Record;
      Success : Boolean;
   begin
      Initialize (Coll);

      Create_VC (1, 100, VC);
      Add_VC (Coll, VC, Success);
      Create_VC (2, 100, VC);
      Add_VC (Coll, VC, Success);
      Create_VC (3, 200, VC);
      Add_VC (Coll, VC, Success);

      Test ("Count VCs for PO 100 is 2", Count_VCs_For_PO (Coll, 100) = 2);
      Test ("Count VCs for PO 200 is 1", Count_VCs_For_PO (Coll, 200) = 1);
   end;

   --  Test 10: All discharged
   declare
      Coll    : VC_Collection;
      VC      : VC_Record;
      Success : Boolean;
   begin
      Initialize (Coll);

      Create_VC (1, 1, VC);
      Update_VC_Status (VC, VC_Valid, 100, 200);
      Add_VC (Coll, VC, Success);

      Test ("All discharged (only valid)", All_Discharged (Coll));

      Create_VC (2, 1, VC);
      Update_VC_Status (VC, VC_Unknown, 100, 200);
      Add_VC (Coll, VC, Success);

      Test ("Not all discharged", not All_Discharged (Coll));
   end;

   --  Test 11: Status and complexity names
   Test ("Status name for Valid", Status_Name (VC_Valid) = "Valid");
   Test ("Status name for Timeout", Status_Name (VC_Timeout) = "Timeout");
   Test ("Complexity name for Trivial",
         Complexity_Name (Trivial) = "Trivial (<10 steps)");

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

end Test_VC_Tracker;
