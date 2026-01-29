--  STUNIR DO-333 Formal Specification Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO;    use Ada.Text_IO;
with Formal_Spec;    use Formal_Spec;
with Spec_Parser;    use Spec_Parser;

procedure Test_Formal_Spec is

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
   Put_Line ("=== Formal Specification Tests ===");
   New_Line;

   --  Test 1: Empty expression
   declare
      E : constant Formal_Expression := Empty_Expression;
   begin
      Test ("Empty expression has zero length", E.Length = 0);
      Test ("Empty expression not verified", not E.Verified);
   end;

   --  Test 2: Create expression
   declare
      E : Formal_Expression;
   begin
      Make_Expression ("X > 0", Precondition, 10, 5, E);
      Test ("Expression kind is Precondition", E.Kind = Precondition);
      Test ("Expression length is 5", E.Length = 5);
      Test ("Expression line is 10", E.Line_Num = 10);
      Test ("Expression column is 5", E.Column = 5);
      Test ("Expression is valid", Is_Valid_Expression (E));
   end;

   --  Test 3: Empty contract
   declare
      C : constant Contract_Spec := Empty_Contract;
   begin
      Test ("Empty contract has zero preconditions", C.Pre_Count = 0);
      Test ("Empty contract has zero postconditions", C.Post_Count = 0);
      Test ("Empty contract has zero invariants", C.Inv_Count = 0);
      Test ("Empty contract is valid", Is_Valid_Contract (C));
      Test ("Empty contract is empty", Is_Empty (C));
   end;

   --  Test 4: Add precondition
   declare
      C       : Contract_Spec := Empty_Contract;
      Success : Boolean;
   begin
      Add_Precondition (C, "X > 0", 10, 1, Success);
      Test ("Add precondition succeeds", Success);
      Test ("Contract has 1 precondition", C.Pre_Count = 1);
      Test ("Contract not empty", not Is_Empty (C));
      Test ("Total conditions is 1", Total_Conditions (C) = 1);
   end;

   --  Test 5: Add postcondition
   declare
      C       : Contract_Spec := Empty_Contract;
      Success : Boolean;
   begin
      Add_Postcondition (C, "Result > X", 15, 1, Success);
      Test ("Add postcondition succeeds", Success);
      Test ("Contract has 1 postcondition", C.Post_Count = 1);
   end;

   --  Test 6: Add invariant
   declare
      C       : Contract_Spec := Empty_Contract;
      Success : Boolean;
   begin
      Add_Invariant (C, "I >= 0", Loop_Invariant, 20, 5, Success);
      Test ("Add invariant succeeds", Success);
      Test ("Contract has 1 invariant", C.Inv_Count = 1);
   end;

   --  Test 7: Multiple conditions
   declare
      C       : Contract_Spec := Empty_Contract;
      Success : Boolean;
   begin
      Add_Precondition (C, "X > 0", 10, 1, Success);
      Add_Precondition (C, "Y > 0", 11, 1, Success);
      Add_Postcondition (C, "Result > 0", 15, 1, Success);
      Add_Invariant (C, "I <= N", Invariant, 20, 5, Success);

      Test ("Contract has 2 preconditions", C.Pre_Count = 2);
      Test ("Contract has 1 postcondition", C.Post_Count = 1);
      Test ("Contract has 1 invariant", C.Inv_Count = 1);
      Test ("Total conditions is 4", Total_Conditions (C) = 4);
   end;

   --  Test 8: Mark verified
   declare
      E : Formal_Expression;
   begin
      Make_Expression ("X > 0", Assertion, 5, 1, E);
      Test ("Expression not verified initially", not E.Verified);
      Mark_Verified (E);
      Test ("Expression verified after marking", E.Verified);
   end;

   --  Test 9: Contains Pre
   Test ("Contains_Pre finds Pre =>",
         Contains_Pre ("   with Pre => X > 0"));
   Test ("Contains_Pre finds Precondition",
         Contains_Pre ("   pragma Precondition (X > 0)"));

   --  Test 10: Contains Post
   Test ("Contains_Post finds Post =>",
         Contains_Post ("   Post => Result > 0"));
   Test ("Contains_Post finds Postcondition",
         Contains_Post ("   pragma Postcondition"));

   --  Test 11: Contains Invariant
   Test ("Contains_Invariant finds Loop_Invariant",
         Contains_Invariant ("      pragma Loop_Invariant (I >= 0)"));
   Test ("Contains_Invariant finds Type_Invariant",
         Contains_Invariant ("   Type_Invariant => Is_Valid"));

   --  Test 12: Ghost line detection
   Test ("Is_Ghost_Line finds Ghost",
         Is_Ghost_Line ("   X : Integer with Ghost;"));
   Test ("Is_Ghost_Line finds pragma Ghost",
         Is_Ghost_Line ("pragma Ghost;"));

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

end Test_Formal_Spec;
