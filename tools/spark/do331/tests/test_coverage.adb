--  STUNIR DO-331 Coverage Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;
with Coverage; use Coverage;
with Coverage_Analysis; use Coverage_Analysis;

procedure Test_Coverage is
   Points    : Coverage_Points;
   Test_Pass : Natural := 0;
   Test_Fail : Natural := 0;
   
   procedure Assert (Condition : Boolean; Test_Name : String) is
   begin
      if Condition then
         Put_Line ("  [PASS] " & Test_Name);
         Test_Pass := Test_Pass + 1;
      else
         Put_Line ("  [FAIL] " & Test_Name);
         Test_Fail := Test_Fail + 1;
      end if;
   end Assert;
   
begin
   Put_Line ("Coverage Tests");
   Put_Line ("==============");
   
   --  Create coverage container
   Points := Create_Coverage;
   Assert (Points.Point_Count = 0, "Container created empty");
   
   --  Add coverage points
   Add_Point (
      Container => Points,
      Kind      => Entry_Coverage,
      Element   => 1,
      Path      => "Module::Function1",
      Point_ID  => "CP_ENTRY_1"
   );
   Assert (Points.Point_Count = 1, "Entry point added");
   
   Add_Point (
      Container => Points,
      Kind      => Decision_Coverage,
      Element   => 1,
      Path      => "Module::Function1::if_1",
      Point_ID  => "CP_DEC_1_T"
   );
   Assert (Points.Point_Count = 2, "Decision point added");
   
   Add_Point (
      Container => Points,
      Kind      => Exit_Coverage,
      Element   => 1,
      Path      => "Module::Function1",
      Point_ID  => "CP_EXIT_1"
   );
   Assert (Points.Point_Count = 3, "Exit point added");
   
   --  Test lookups
   Assert (Point_Exists (Points, "CP_ENTRY_1"), "Entry point exists");
   Assert (not Point_Exists (Points, "CP_NONEXISTENT"), "Non-existent point");
   
   --  Test DAL requirements
   Assert (Is_Required (DAL_A, MCDC_Coverage), "DAL A requires MC/DC");
   Assert (not Is_Required (DAL_B, MCDC_Coverage), "DAL B doesn't require MC/DC");
   Assert (Is_Required (DAL_C, State_Coverage), "DAL C requires state coverage");
   
   --  Test coverage status
   Mark_Covered (Points, "CP_ENTRY_1");
   declare
      Point : constant Coverage_Point := Get_Point_By_ID (Points, "CP_ENTRY_1");
   begin
      Assert (Point.Covered, "Point marked as covered");
   end;
   
   --  Test analysis
   declare
      Result : constant Analysis_Result := Analyze (Points);
   begin
      Assert (Result.Stats.Total_Points = 3, "Analysis counts total points");
      Assert (Result.Stats.Covered_Count = 1, "Analysis counts covered points");
      Assert (Result.Stats.Entry_Points = 1, "Analysis counts entry points");
      Assert (Result.Stats.Decision_Points = 1, "Analysis counts decision points");
   end;
   
   --  Test DAL-specific analysis
   declare
      DAL_Points : constant Coverage_Point_Array := Get_Points_For_DAL (Points, DAL_C);
   begin
      Assert (DAL_Points'Length > 0, "DAL C points retrieved");
   end;
   
   --  Test type prefix
   Assert (Get_Type_Prefix (State_Coverage) = "CP_STATE", "State prefix correct");
   Assert (Get_Type_Prefix (MCDC_Coverage) = "CP_MCDC", "MC/DC prefix correct");
   
   --  Summary
   Put_Line ("");
   Put_Line ("Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
end Test_Coverage;
