--  STUNIR DO-332 Dispatch Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;

with OOP_Types; use OOP_Types;
with OOP_Analysis; use OOP_Analysis;
with Dispatch_Analyzer; use Dispatch_Analyzer;
with VTable_Builder; use VTable_Builder;

procedure Test_Dispatch is

   Test_Classes : constant Class_Array (1 .. 3) := (
      1 => (ID => 1, Name => (1 => 'B', 2 => 'a', 3 => 's', 4 => 'e', others => ' '), Name_Length => 4,
            Kind => Abstract_Class, Parent_Count => 0, Method_Count => 1, Field_Count => 0,
            Is_Root => True, Is_Abstract => True, Is_Final => False,
            Inheritance_Depth => 0, Line_Number => 1, File_Path => (others => ' '), File_Path_Length => 0),
      2 => (ID => 2, Name => (1 => 'D', 2 => '1', others => ' '), Name_Length => 2,
            Kind => Regular_Class, Parent_Count => 1, Method_Count => 1, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => False,
            Inheritance_Depth => 1, Line_Number => 10, File_Path => (others => ' '), File_Path_Length => 0),
      3 => (ID => 3, Name => (1 => 'D', 2 => '2', others => ' '), Name_Length => 2,
            Kind => Final_Class, Parent_Count => 1, Method_Count => 1, Field_Count => 0,
            Is_Root => False, Is_Abstract => False, Is_Final => True,
            Inheritance_Depth => 1, Line_Number => 20, File_Path => (others => ' '), File_Path_Length => 0)
   );

   Test_Methods : constant Method_Array (1 .. 3) := (
      1 => (ID => 1, Name => (1 => 'd', 2 => 'o', 3 => 'I', 4 => 't', others => ' '), Name_Length => 4,
            Owning_Class => 1, Kind => Abstract_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => False, Override_Of => Null_Method_ID,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 2),
      2 => (ID => 2, Name => (1 => 'd', 2 => 'o', 3 => 'I', 4 => 't', others => ' '), Name_Length => 4,
            Owning_Class => 2, Kind => Virtual_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 1,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 12),
      3 => (ID => 3, Name => (1 => 'd', 2 => 'o', 3 => 'I', 4 => 't', others => ' '), Name_Length => 4,
            Owning_Class => 3, Kind => Final_Method, Visibility => V_Public,
            Parameter_Count => 0, Has_Override => True, Override_Of => 1,
            Is_Covariant => True, Is_Contravariant => False, Line_Number => 22)
   );

   Test_Links : constant Inheritance_Array (1 .. 2) := (
      1 => (Child_ID => 2, Parent_ID => 1, Is_Virtual => False, Is_Interface => False, Link_Index => 1),
      2 => (Child_ID => 3, Parent_ID => 1, Is_Virtual => False, Is_Interface => False, Link_Index => 1)
   );

   Test_Sites : constant Dispatch_Site_Array (1 .. 1) := (
      1 => (Site_ID => 1, Location_File => (others => ' '), File_Length => 0,
            Location_Line => 50, Receiver_Type => 1,
            Method_Name => (1 => 'd', 2 => 'o', 3 => 'I', 4 => 't', others => ' '),
            Method_Name_Len => 4, Is_Super_Call => False, Is_Interface => False)
   );

   Results : Site_Analysis_Array (1 .. 1);
   Summary : Dispatch_Summary;
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
   Put_Line ("=== Dispatch Analyzer Tests ===");
   New_Line;

   --  Test 1: Target resolution
   Put_Line ("Test 1: Target Resolution");
   declare
      Targets : constant Target_Array := 
         Resolve_Targets (Test_Sites (1), Test_Classes, Test_Methods, Test_Links);
   begin
      Check ("Found targets", Targets'Length > 0);
      Check ("Found 2 concrete targets", Targets'Length = 2);
   end;
   New_Line;

   --  Test 2: Site analysis
   Put_Line ("Test 2: Site Analysis");
   declare
      Analysis : constant Site_Analysis := 
         Analyze_Site (Test_Sites (1), Test_Classes, Test_Methods, Test_Links);
   begin
      Check ("Site is bounded", Analysis.Is_Bounded);
      Check ("Target count correct", Analysis.Target_Count = 2);
      Check ("Not devirtualizable (2 targets)", not Analysis.Is_Devirtualizable);
   end;
   New_Line;

   --  Test 3: VTable construction
   Put_Line ("Test 3: VTable Construction");
   declare
      VT : constant VTable := 
         Build_VTable (Test_Classes (2), Test_Classes, Test_Methods, Test_Links);
   begin
      Check ("VTable built", VT.Class_ID = 2);
      Check ("Has entries", VT.Entry_Count > 0);
      Check ("No abstract methods", not VT.Has_Abstract);
   end;
   New_Line;

   --  Test 4: Full dispatch analysis
   Put_Line ("Test 4: Full Dispatch Analysis");
   Analyze_All_Dispatch (Test_Sites, Test_Classes, Test_Methods, Test_Links,
                         Results, Summary, Success);
   Check ("Analysis completed", Success);
   Check ("All sites bounded", Summary.Bounded_Sites = Summary.Total_Sites);
   Check ("Max targets tracked", Summary.Max_Targets >= 2);
   New_Line;

   --  Test 5: Devirtualization detection
   Put_Line ("Test 5: Devirtualization Detection");
   declare
      Single_Site : constant Dispatch_Site := (
         Site_ID => 2, Location_File => (others => ' '), File_Length => 0,
         Location_Line => 60, Receiver_Type => 3,  --  Final class
         Method_Name => (1 => 'd', 2 => 'o', 3 => 'I', 4 => 't', others => ' '),
         Method_Name_Len => 4, Is_Super_Call => False, Is_Interface => False);
   begin
      Check ("Final class devirtualizable", 
             Can_Devirtualize (Single_Site, Test_Classes, Test_Methods, Test_Links));
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

end Test_Dispatch;
