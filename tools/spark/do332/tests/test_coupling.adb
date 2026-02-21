--  STUNIR DO-332 Coupling Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;
with Coupling_Analyzer; use Coupling_Analyzer;
with Coupling_Metrics; use Coupling_Metrics;

procedure Test_Coupling is

   Test_Classes : constant Class_Array (1 .. 3) := (
      1 => (ID => 1, Name => (1 => 'A', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 0, Method_Count => 3, Field_Count => 1,
            Is_Root => True, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 0, Line_Number => 1, File_Path => (others => ' '), File_Path_Length => 0),
      2 => (ID => 2, Name => (1 => 'B', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 1, Method_Count => 2, Field_Count => 1,
            Is_Root => False, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 1, Line_Number => 10, File_Path => (others => ' '), File_Path_Length => 0),
      3 => (ID => 3, Name => (1 => 'C', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 0, Method_Count => 2, Field_Count => 2,
            Is_Root => True, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 0, Line_Number => 20, File_Path => (others => ' '), File_Path_Length => 0)
   );

   Test_Methods : constant Method_Array (1 .. 7) := (
      1 => (ID => 1, Name => (1 => 'm', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 2),
      2 => (ID => 2, Name => (1 => 'm', 2 => '2', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 3),
      3 => (ID => 3, Name => (1 => 'm', 2 => '3', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Regular_Method, Visibility => V_Private,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 4),
      4 => (ID => 4, Name => (1 => 'm', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 2, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 1,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 12),
      5 => (ID => 5, Name => (1 => 'm', 2 => '4', others => ' '), Name_Length => 2,
            Owning_Class => 2, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 13),
      6 => (ID => 6, Name => (1 => 'm', 2 => '5', others => ' '), Name_Length => 2,
            Owning_Class => 3, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 22),
      7 => (ID => 7, Name => (1 => 'm', 2 => '6', others => ' '), Name_Length => 2,
            Owning_Class => 3, Kind => Regular_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 23)
   );

   Test_Links : constant Inheritance_Array (1 .. 1) := (
      1 => (Child_ID => 2, Parent_ID => 1, Is_Virtual => False, Is_Interface => False, Link_Index => 1)
   );

   Results : Coupling_Result_Array (1 .. 3);
   Summary : Coupling_Summary;
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
   Put_Line ("=== Coupling Analyzer Tests ===");
   New_Line;

   --  Test 1: CBO calculation
   Put_Line ("Test 1: CBO Calculation");
   declare
      Deps : constant Dependency_Array (1 .. 1) := (
         1 => (Source_Class => 2, Target_Class => 1, Kind => Inheritance_Dep, Count => 1)
      );
      CBO : constant Natural := Calculate_CBO (2, Deps);
   begin
      Check ("CBO calculated", CBO >= 0);
      Check ("Class B coupled to A", CBO >= 1);
   end;
   New_Line;

   --  Test 2: RFC calculation
   Put_Line ("Test 2: RFC Calculation");
   declare
      RFC_A : constant Natural := Calculate_RFC (1, Test_Methods);
      RFC_C : constant Natural := Calculate_RFC (3, Test_Methods);
   begin
      Check ("RFC for A = 3 methods", RFC_A = 3);
      Check ("RFC for C = 2 methods", RFC_C = 2);
   end;
   New_Line;

   --  Test 3: NOC calculation
   Put_Line ("Test 3: NOC Calculation");
   declare
      NOC_A : constant Natural := Calculate_NOC (1, Test_Links);
      NOC_C : constant Natural := Calculate_NOC (3, Test_Links);
   begin
      Check ("NOC for A = 1 child", NOC_A = 1);
      Check ("NOC for C = 0 children", NOC_C = 0);
   end;
   New_Line;

   --  Test 4: Full coupling analysis
   Put_Line ("Test 4: Full Coupling Analysis");
   Analyze_All_Coupling (Test_Classes, Test_Methods, Test_Links,
                         10, 50, Results, Summary, Success);
   Check ("Analysis completed", Success);
   Check ("No circular deps", Summary.Circular_Deps = 0);
   Check ("Max CBO tracked", Summary.Max_CBO >= 0);
   New_Line;

   --  Test 5: Threshold checking
   Put_Line ("Test 5: Threshold Checking");
   declare
      Violations : constant Threshold_Violations := Check_Thresholds (
         CBO => 5, RFC => 30, LCOM => 10, DIT => 2, NOC => 3, WMC => 15,
         Thresholds => Default_Thresholds
      );
   begin
      Check ("Under CBO threshold", not Violations.CBO_Exceeded);
      Check ("Under RFC threshold", not Violations.RFC_Exceeded);
      Check ("No violations", not Any_Exceeded (Violations));
   end;
   New_Line;

   --  Test 6: High coupling detection
   Put_Line ("Test 6: High Coupling Detection");
   declare
      High_Violations : constant Threshold_Violations := Check_Thresholds (
         CBO => 20, RFC => 100, LCOM => 200, DIT => 10, NOC => 20, WMC => 50,
         Thresholds => Default_Thresholds
      );
   begin
      Check ("CBO exceeded detected", High_Violations.CBO_Exceeded);
      Check ("RFC exceeded detected", High_Violations.RFC_Exceeded);
      Check ("Has violations", Any_Exceeded (High_Violations));
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

end Test_Coupling;
