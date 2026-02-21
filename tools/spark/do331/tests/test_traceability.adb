--  STUNIR DO-331 Traceability Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;
with Traceability; use Traceability;

procedure Test_Traceability is
   Matrix    : Trace_Matrix;
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
   Put_Line ("Traceability Tests");
   Put_Line ("==================");
   
   --  Create matrix
   Matrix := Create_Matrix;
   Assert (Matrix.Entry_Count = 0, "Matrix created empty");
   
   --  Add traces
   Add_Trace (
      Matrix    => Matrix,
      Kind      => Trace_IR_To_Model,
      Source_ID => 1,
      Src_Path  => "module.function1",
      Target_ID => 100,
      Tgt_Path  => "Module::Function1",
      Rule      => Rule_Function_To_Action
   );
   Assert (Matrix.Entry_Count = 1, "First trace added");
   
   Add_Trace (
      Matrix    => Matrix,
      Kind      => Trace_IR_To_Model,
      Source_ID => 2,
      Src_Path  => "module.function2",
      Target_ID => 101,
      Tgt_Path  => "Module::Function2",
      Rule      => Rule_Function_To_Action
   );
   Assert (Matrix.Entry_Count = 2, "Second trace added");
   
   --  Test lookups
   Assert (Has_Trace (Matrix, 1), "Source 1 has trace");
   Assert (Has_Trace (Matrix, 100), "Target 100 has trace");
   Assert (not Has_Trace (Matrix, 999), "Non-existent ID has no trace");
   
   --  Test forward traces
   declare
      Forward : constant Trace_Entry_Array := Get_Forward_Traces (Matrix, 1);
   begin
      Assert (Forward'Length = 1, "Forward trace found");
   end;
   
   --  Test backward traces
   declare
      Backward : constant Trace_Entry_Array := Get_Backward_Traces (Matrix, 100);
   begin
      Assert (Backward'Length = 1, "Backward trace found");
   end;
   
   --  Test hash setting
   Set_IR_Hash (Matrix, "abc123");
   Assert (Matrix.IR_Hash_Len = 6, "IR hash set");
   
   Set_Model_Hash (Matrix, "def456");
   Assert (Matrix.Model_Hash_Len = 6, "Model hash set");
   
   --  Test validation
   Assert (Validate_Matrix (Matrix), "Matrix is valid");
   
   --  Test completeness
   declare
      IDs : constant Element_ID_Array := (1 => 1, 2 => 2);
   begin
      Assert (Check_Completeness (Matrix, IDs), "Completeness check passes");
   end;
   
   --  Test gap analysis
   declare
      IDs    : constant Element_ID_Array := (1 => 1, 2 => 2, 3 => 999);
      Report : constant Gap_Report := Analyze_Gaps (Matrix, IDs);
   begin
      Assert (Report.Total_IR_Elements = 3, "Gap analysis counts total");
      Assert (Report.Traced_Elements = 2, "Gap analysis counts traced");
      Assert (Report.Missing_Traces = 1, "Gap analysis finds missing");
      Assert (not Report.Is_Complete, "Gap analysis detects incompleteness");
   end;
   
   --  Summary
   Put_Line ("");
   Put_Line ("Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
end Test_Traceability;
