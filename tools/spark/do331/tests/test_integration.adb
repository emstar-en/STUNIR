--  STUNIR DO-331 Integration Tests
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;
with SysML_Emitter; use SysML_Emitter;
with Traceability; use Traceability;
with Trace_Matrix;
with Coverage; use Coverage;
with Coverage_Analysis; use Coverage_Analysis;

procedure Test_Integration is
   Container     : Model_Container;
   Storage       : Element_Storage;
   Root_ID       : Element_ID;
   Action1_ID    : Element_ID;
   Action2_ID    : Element_ID;
   State1_ID     : Element_ID;
   State2_ID     : Element_ID;
   Trans_ID      : Element_ID;
   Trace         : Traceability.Trace_Matrix;
   Cov_Points    : Coverage_Points;
   Buffer        : Output_Buffer;
   Emit_Result   : Emitter_Result;
   Cov_Result    : Analysis_Result;
   Test_Pass     : Natural := 0;
   Test_Fail     : Natural := 0;
   
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
   Put_Line ("Integration Tests");
   Put_Line ("=================");
   Put_Line ("");
   Put_Line ("Testing complete DO-331 transformation pipeline...");
   Put_Line ("");
   
   --  Initialize
   Reset_ID_Generator;
   Initialize_Storage (Storage);
   Container := Create_Container;
   Trace := Create_Matrix;
   Cov_Points := Create_Coverage;
   
   --  STEP 1: Create model from IR
   Put_Line ("Step 1: IR to Model transformation");
   
   Transform_Module (
      Module_Name => "altitude_controller",
      IR_Hash     => "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      Options     => Default_Options,
      Container   => Container,
      Storage     => Storage,
      Root_ID     => Root_ID
   );
   Assert (Root_ID /= Null_Element_ID, "Module transformed");
   
   --  Add traceability
   Add_Trace (
      Matrix    => Trace,
      Kind      => Trace_IR_To_Model,
      Source_ID => 1,
      Src_Path  => "altitude_controller",
      Target_ID => Root_ID,
      Tgt_Path  => "AltitudeController",
      Rule      => Rule_Module_To_Package
   );
   
   --  STEP 2: Transform functions
   Put_Line ("Step 2: Function transformation");
   
   Transform_Function (
      Func_Name    => "check_altitude",
      Parent_ID    => Root_ID,
      Has_Inputs   => True,
      Has_Outputs  => True,
      Input_Count  => 1,
      Output_Count => 1,
      Storage      => Storage,
      Action_ID    => Action1_ID
   );
   Assert (Action1_ID /= Null_Element_ID, "Function 1 transformed");
   
   Transform_Function (
      Func_Name    => "adjust_throttle",
      Parent_ID    => Root_ID,
      Has_Inputs   => True,
      Has_Outputs  => True,
      Input_Count  => 2,
      Output_Count => 1,
      Storage      => Storage,
      Action_ID    => Action2_ID
   );
   Assert (Action2_ID /= Null_Element_ID, "Function 2 transformed");
   
   --  Add coverage points for functions
   Add_Point (Cov_Points, Entry_Coverage, Action1_ID, 
              "AltitudeController::CheckAltitude", "CP_ENTRY_1");
   Add_Point (Cov_Points, Decision_Coverage, Action1_ID,
              "AltitudeController::CheckAltitude::if_1", "CP_DEC_1_T");
   Add_Point (Cov_Points, Exit_Coverage, Action1_ID,
              "AltitudeController::CheckAltitude", "CP_EXIT_1");
   Add_Point (Cov_Points, Entry_Coverage, Action2_ID,
              "AltitudeController::AdjustThrottle", "CP_ENTRY_2");
   Add_Point (Cov_Points, Exit_Coverage, Action2_ID,
              "AltitudeController::AdjustThrottle", "CP_EXIT_2");
   Assert (Cov_Points.Point_Count = 5, "Coverage points added");
   
   --  STEP 3: Transform state machine
   Put_Line ("Step 3: State machine transformation");
   
   Add_State (
      State_Name => "idle",
      Parent_ID  => Root_ID,
      Is_Initial => True,
      Is_Final   => False,
      Storage    => Storage,
      State_ID   => State1_ID
   );
   Assert (State1_ID /= Null_Element_ID, "State 1 created");
   
   Add_State (
      State_Name => "active",
      Parent_ID  => Root_ID,
      Is_Initial => False,
      Is_Final   => False,
      Storage    => Storage,
      State_ID   => State2_ID
   );
   Assert (State2_ID /= Null_Element_ID, "State 2 created");
   
   Add_Transition (
      Trans_Name => "activate",
      Source_ID  => State1_ID,
      Target_ID  => State2_ID,
      Has_Guard  => True,
      Guard_Expr => "altitude > 1000",
      Parent_ID  => Root_ID,
      Storage    => Storage,
      Trans_ID   => Trans_ID
   );
   Assert (Trans_ID /= Null_Element_ID, "Transition created");
   
   --  Add state coverage points
   Add_Point (Cov_Points, State_Coverage, State1_ID,
              "AltitudeController::Idle", "CP_STATE_1");
   Add_Point (Cov_Points, State_Coverage, State2_ID,
              "AltitudeController::Active", "CP_STATE_2");
   Add_Point (Cov_Points, Transition_Coverage, Trans_ID,
              "AltitudeController::activate", "CP_TRANS_1");
   Assert (Cov_Points.Point_Count = 8, "State coverage points added");
   
   --  STEP 4: Emit SysML 2.0
   Put_Line ("Step 4: SysML 2.0 emission");
   
   Emit_Model (
      Container => Container,
      Storage   => Storage,
      Options   => Default_Emitter_Options,
      Buffer    => Buffer,
      Result    => Emit_Result
   );
   
   Assert (Emit_Result.Status = Emit_Success, "Emission successful");
   Assert (Emit_Result.Output_Length > 100, "Substantial output generated");
   Assert (Emit_Result.Line_Count > 10, "Multiple lines generated");
   
   --  STEP 5: Coverage analysis
   Put_Line ("Step 5: Coverage analysis");
   
   Cov_Result := Analyze_For_DAL (Cov_Points, DAL_B);
   
   Assert (Cov_Result.Stats.Total_Points = 8, "Total points counted");
   Assert (Cov_Result.Stats.State_Points = 2, "State points counted");
   Assert (Cov_Result.Stats.Transition_Points = 1, "Transition points counted");
   Assert (Cov_Result.Stats.Entry_Points = 2, "Entry points counted");
   Assert (Cov_Result.Stats.Exit_Points = 2, "Exit points counted");
   Assert (Cov_Result.Stats.Decision_Points = 1, "Decision points counted");
   
   --  STEP 6: Traceability completeness
   Put_Line ("Step 6: Traceability verification");
   
   Assert (Trace.Entry_Count >= 1, "Trace entries exist");
   Assert (Validate_Matrix (Trace), "Trace matrix valid");
   
   --  Final summary
   Put_Line ("");
   Put_Line ("Pipeline Summary:");
   Put_Line ("  Elements created: " & Natural'Image (Storage.Count));
   Put_Line ("  Trace entries: " & Natural'Image (Trace.Entry_Count));
   Put_Line ("  Coverage points: " & Natural'Image (Cov_Points.Point_Count));
   Put_Line ("  Output size: " & Natural'Image (Emit_Result.Output_Length) & " bytes");
   Put_Line ("");
   Put_Line ("Test Results: " & Natural'Image (Test_Pass) & " passed," &
             Natural'Image (Test_Fail) & " failed");
   
   if Test_Fail = 0 then
      Put_Line ("");
      Put_Line ("*** ALL INTEGRATION TESTS PASSED ***");
   end if;
end Test_Integration;
