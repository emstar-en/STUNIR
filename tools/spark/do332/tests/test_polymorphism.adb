--  STUNIR DO-332 Polymorphism Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;
with Polymorphism_Verifier; use Polymorphism_Verifier;
with Substitutability; use Substitutability;

procedure Test_Polymorphism is

   Test_Classes : constant Class_Array (1 .. 2) := (
      1 => (ID => 1, Name => (1 => 'P', others => ' '), Name_Length => 1,
            Kind => Abstract_Class, Parent_Count => 0, Method_Count => 2, Field_Count => 0,
            Is_Root => True, Is_Abstract => True, Is_Final => False,
            Inheritance_Depth => 0, Line_Number => 1, File_Path => (others => ' '), File_Path_Length => 0),
      2 => (ID => 2, Name => (1 => 'C', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 1, Method_Count => 2, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 1, Line_Number => 10, File_Path => (others => ' '), File_Path_Length => 0)
   );

   Test_Methods : constant Method_Array (1 .. 4) := (
      1 => (ID => 1, Name => (1 => 'v', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Abstract_Method, Visibility => V_Public,
            Parameter_Count => 1, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 2),
      2 => (ID => 2, Name => (1 => 'v', 2 => '2', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 3),
      3 => (ID => 3, Name => (1 => 'v', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 2, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 1, Has_Override => True, Override_Of => 1,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 12),
      4 => (ID => 4, Name => (1 => 'v', 2 => '2', others => ' '), Name_Length => 2,
            Owning_Class => 2, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 2,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 13)
   );

   Test_Links : constant Inheritance_Array (1 .. 1) := (
      1 => (Child_ID => 2, Parent_ID => 1, Is_Virtual => False, Is_Interface => False, Link_Index => 1)
   );

   Results : Polymorphism_Result_Array (1 .. 2);
   Success : Boolean;
   Test_Count : Natural := 0;
   Pass_Count : Natural := 0;

   procedure Check (Name : String; Condition : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Condition then
         Pass_Count := Pass_Count + 1;
         Put_Line ("  PASS: " & Name);
      else
         Put_Line ("  FAIL: " & Name);
      end if;
   end Check;

begin
   Put_Line ("=== Polymorphism Verifier Tests ===");
   New_Line;

   --  Test 1: Virtual method scanning
   Put_Line ("Test 1: Virtual Method Scanning");
   declare
      Virtuals : constant Virtual_Method_Array := Scan_Virtual_Methods (Test_Methods);
   begin
      Check ("Found virtual methods", Virtuals'Length > 0);
      Check ("Abstract methods detected", Virtuals (1).Is_Abstract);
   end;
   New_Line;

   --  Test 2: Type counting
   Put_Line ("Test 2: Possible Type Counting");
   declare
      Type_Count : constant Natural := Count_Possible_Types (1, Test_Links);
   begin
      Check ("Parent has 2 possible types", Type_Count = 2);
      Check ("Child has 1 possible type", Count_Possible_Types (2, Test_Links) = 1);
   end;
   New_Line;

   --  Test 3: Full verification
   Put_Line ("Test 3: Full Polymorphism Verification");
   Verify_All_Polymorphism (Test_Classes, Test_Methods, Test_Links, Results, Success);
   Check ("Verification completed", Success);
   Check ("Abstract class has virtuals", Results (1).Virtual_Methods > 0);
   Check ("Type safe", Results (2).Type_Safe);
   New_Line;

   --  Test 4: LSP checking
   Put_Line ("Test 4: LSP Compliance Checking");
   declare
      Result : constant LSP_Check_Result := Check_LSP (Test_Methods (1), Test_Methods (3));
   begin
      Check ("Override is LSP compliant", Result = LSP_Compliant);
   end;
   New_Line;

   --  Test 5: Substitutability
   Put_Line ("Test 5: Substitutability Analysis");
   declare
      Summary : Substitutability_Summary;
      Errors  : Boolean;
   begin
      Find_LSP_Violations (Test_Classes, Test_Methods, Test_Links, Summary, Errors);
      Check ("No LSP errors found", not Errors);
      Check ("Overrides checked", Summary.Total_Checked > 0);
   end;
   New_Line;

   --  Summary
   Put_Line ("===================================");
   Put_Line ("Tests: " & Natural'Image (Test_Count) & 
             " | Passed: " & Natural'Image (Pass_Count) &
             " | Failed: " & Natural'Image (Test_Count - Pass_Count));

   if Pass_Count = Test_Count then
      Put_Line ("All tests PASSED!");
   end if;

end Test_Polymorphism;
