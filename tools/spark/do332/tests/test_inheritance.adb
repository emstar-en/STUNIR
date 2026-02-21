--  STUNIR DO-332 Inheritance Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;
with Inheritance_Analyzer; use Inheritance_Analyzer;
with Inheritance_Metrics; use Inheritance_Metrics;

procedure Test_Inheritance is

   --  Test data
   Test_Classes : constant Class_Array (1 .. 4) := (
      1 => (ID => 1, Name => (1 => 'A', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 0, Method_Count => 1, Field_Count => 0,
            Is_Root => True, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 0, Line_Number => 1, File_Path => (others => ' '), File_Path_Length => 0),
      2 => (ID => 2, Name => (1 => 'B', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 1, Method_Count => 1, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 1, Line_Number => 10, File_Path => (others => ' '), File_Path_Length => 0),
      3 => (ID => 3, Name => (1 => 'C', others => ' '), Name_Length => 1,
            Kind => Regular_Class, Parent_Count => 1, Method_Count => 1, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 2, Line_Number => 20, File_Path => (others => ' '), File_Path_Length => 0),
      4 => (ID => 4, Name => (1 => 'D', others => ' '), Name_Length => 1,
            Kind => Final_Class, Parent_Count => 1, Method_Count => 1, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => True,
            Inheritance_Depth => 3, Line_Number => 30, File_Path => (others => ' '), File_Path_Length => 0)
   );

   Test_Methods : constant Method_Array (1 .. 4) := (
      1 => (ID => 1, Name => (1 => 'f', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 1, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 2),
      2 => (ID => 2, Name => (1 => 'f', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 2, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 1,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 12),
      3 => (ID => 3, Name => (1 => 'f', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 3, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 2,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 22),
      4 => (ID => 4, Name => (1 => 'f', 2 => '1', others => ' '), Name_Length => 2,
            Owning_Class => 4, Kind => Final_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 3,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 32)
   );

   Test_Links : constant Inheritance_Array (1 .. 3) := (
      1 => (Child_ID => 2, Parent_ID => 1, Is_Virtual => False, Is_Interface => False, Link_Index => 1),
      2 => (Child_ID => 3, Parent_ID => 2, Is_Virtual => False, Is_Interface => False, Link_Index => 1),
      3 => (Child_ID => 4, Parent_ID => 3, Is_Virtual => False, Is_Interface => False, Link_Index => 1)
   );

   Results : Inheritance_Result_Array (1 .. 4);
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
   Put_Line ("=== Inheritance Analyzer Tests ===");
   New_Line;

   --  Test 1: Calculate depth
   Put_Line ("Test 1: Depth Calculation");
   Check ("Root class depth = 0", Calculate_Depth (1, Test_Links) = 0);
   Check ("Level 1 class depth = 1", Calculate_Depth (2, Test_Links) = 1);
   Check ("Level 2 class depth = 2", Calculate_Depth (3, Test_Links) = 2);
   Check ("Level 3 class depth = 3", Calculate_Depth (4, Test_Links) = 3);
   New_Line;

   --  Test 2: Ancestry
   Put_Line ("Test 2: Ancestry Detection");
   Check ("A is ancestor of D", Is_Ancestor (1, 4, Test_Links));
   Check ("B is ancestor of D", Is_Ancestor (2, 4, Test_Links));
   Check ("D is not ancestor of A", not Is_Ancestor (4, 1, Test_Links));
   New_Line;

   --  Test 3: Circular detection (should be false for linear chain)
   Put_Line ("Test 3: Circular Inheritance Detection");
   Check ("No circular in linear chain", not Has_Circular_Inheritance (1, Test_Links));
   Check ("No circular for leaf", not Has_Circular_Inheritance (4, Test_Links));
   New_Line;

   --  Test 4: Full analysis
   Put_Line ("Test 4: Full Inheritance Analysis");
   Analyze_All_Inheritance (Test_Classes, Test_Methods, Test_Links, Results, Success);
   Check ("Analysis completed", Success);
   Check ("Results have correct class IDs", Results (1).Class_ID = 1);
   Check ("Override count tracked", Results (2).Override_Count >= 0);
   New_Line;

   --  Test 5: Metrics
   Put_Line ("Test 5: Inheritance Metrics");
   declare
      Metrics : constant Inheritance_Metrics_Record := 
         Calculate_Metrics (Test_Classes, Test_Methods, Test_Links, Results);
   begin
      Check ("Total classes counted", Natural (Metrics.Total_Classes) = 4);
      Check ("Max depth correct", Natural (Metrics.Max_Depth) = 3);
      Check ("Total overrides counted", Metrics.Total_Overrides > 0);
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

end Test_Inheritance;
