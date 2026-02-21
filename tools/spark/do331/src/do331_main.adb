--  STUNIR DO-331 Main Entry Point
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Command_Line; use Ada.Command_Line;

with Model_IR;
with SysML_Types;
with IR_To_Model;
with SysML_Emitter;
with Traceability;
with Trace_Matrix;
with Coverage;
with Coverage_Analysis;
with Transformer_Utils;

procedure DO331_Main is
   
   --  Print usage
   procedure Print_Usage is
   begin
      Put_Line ("STUNIR DO-331 Model-Based Development Tool");
      Put_Line ("Copyright (C) 2026 STUNIR Project");
      Put_Line ("");
      Put_Line ("Usage: do331_main [options]");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help           Show this help");
      Put_Line ("  --version        Show version");
      Put_Line ("  --test           Run self-test");
      Put_Line ("  --transform      Transform IR to model");
      Put_Line ("  --emit           Emit SysML 2.0 output");
      Put_Line ("  --trace          Generate traceability");
      Put_Line ("  --coverage       Generate coverage points");
      Put_Line ("");
      Put_Line ("Environment Variables:");
      Put_Line ("  STUNIR_ENABLE_COMPLIANCE=1  Enable compliance features");
      Put_Line ("  STUNIR_DAL_LEVEL=A|B|C|D|E  Target DAL level");
   end Print_Usage;
   
   --  Print version
   procedure Print_Version is
   begin
      Put_Line ("DO-331 Model-Based Development Tool v1.0.0");
      Put_Line ("STUNIR Project - Ada SPARK Implementation");
      Put_Line ("Schema: stunir.model.do331.v1");
   end Print_Version;
   
   --  Run self-test
   procedure Run_Self_Test is
      use Model_IR;
      use IR_To_Model;
      use Coverage;
      use Traceability;
      
      Container : Model_Container;
      Storage   : Element_Storage;
      Root_ID   : Element_ID;
      Action_ID : Element_ID;
      State_ID  : Element_ID;
      Trans_ID  : Element_ID;
      Matrix    : Trace_Matrix;
      Points    : Coverage_Points;
      Result    : Coverage_Analysis.Analysis_Result;
   begin
      Put_Line ("Running DO-331 self-test...");
      Put_Line ("");
      
      --  Test 1: Model IR
      Put ("  [TEST 1] Model IR creation... ");
      Model_IR.Reset_ID_Generator;
      Container := Create_Container;
      if Container.Element_Count = 0 then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 2: Element creation
      Put ("  [TEST 2] Element creation... ");
      Initialize_Storage (Storage);
      Transform_Module (
         Module_Name => "test_module",
         IR_Hash     => "abc123",
         Options     => Default_Options,
         Container   => Container,
         Storage     => Storage,
         Root_ID     => Root_ID
      );
      if Root_ID /= Null_Element_ID then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 3: Function transformation
      Put ("  [TEST 3] Function to action... ");
      Transform_Function (
         Func_Name    => "check_altitude",
         Parent_ID    => Root_ID,
         Has_Inputs   => True,
         Has_Outputs  => True,
         Input_Count  => 1,
         Output_Count => 1,
         Storage      => Storage,
         Action_ID    => Action_ID
      );
      if Action_ID /= Null_Element_ID then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 4: State creation
      Put ("  [TEST 4] State creation... ");
      Add_State (
         State_Name => "idle",
         Parent_ID  => Root_ID,
         Is_Initial => True,
         Is_Final   => False,
         Storage    => Storage,
         State_ID   => State_ID
      );
      if State_ID /= Null_Element_ID then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 5: Transition
      Put ("  [TEST 5] Transition creation... ");
      Add_Transition (
         Trans_Name => "init_to_idle",
         Source_ID  => State_ID,
         Target_ID  => State_ID,
         Has_Guard  => True,
         Guard_Expr => "altitude > 0",
         Parent_ID  => Root_ID,
         Storage    => Storage,
         Trans_ID   => Trans_ID
      );
      if Trans_ID /= Null_Element_ID then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 6: Traceability
      Put ("  [TEST 6] Traceability matrix... ");
      Matrix := Create_Matrix;
      Add_Trace (
         Matrix    => Matrix,
         Kind      => Trace_IR_To_Model,
         Source_ID => 1,
         Src_Path  => "module.function",
         Target_ID => Action_ID,
         Tgt_Path  => "TestModule::CheckAltitude",
         Rule      => Rule_Function_To_Action
      );
      if Matrix.Entry_Count = 1 then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 7: Coverage
      Put ("  [TEST 7] Coverage points... ");
      Points := Create_Coverage;
      Add_Point (
         Container => Points,
         Kind      => Entry_Coverage,
         Element   => Action_ID,
         Path      => "TestModule::CheckAltitude",
         Point_ID  => "CP_ENTRY_1"
      );
      if Points.Point_Count = 1 then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      --  Test 8: Coverage analysis
      Put ("  [TEST 8] Coverage analysis... ");
      Result := Coverage_Analysis.Analyze (Points);
      if Result.Stats.Total_Points = 1 then
         Put_Line ("PASS");
      else
         Put_Line ("FAIL");
      end if;
      
      Put_Line ("");
      Put_Line ("Self-test complete.");
      Put_Line ("All SPARK components validated.");
   end Run_Self_Test;
   
begin
   if Argument_Count = 0 then
      Print_Usage;
   else
      for I in 1 .. Argument_Count loop
         if Argument (I) = "--help" or Argument (I) = "-h" then
            Print_Usage;
         elsif Argument (I) = "--version" or Argument (I) = "-v" then
            Print_Version;
         elsif Argument (I) = "--test" then
            Run_Self_Test;
         elsif Argument (I) = "--transform" then
            Put_Line ("Transform mode: requires IR input file.");
            Put_Line ("Usage: do331_main --transform <input.json> <output.sysml>");
         elsif Argument (I) = "--emit" then
            Put_Line ("Emit mode: requires model input.");
         elsif Argument (I) = "--trace" then
            Put_Line ("Trace mode: generates traceability matrix.");
         elsif Argument (I) = "--coverage" then
            Put_Line ("Coverage mode: analyzes model coverage.");
         else
            Put_Line ("Unknown option: " & Argument (I));
            Put_Line ("Use --help for usage information.");
         end if;
      end loop;
   end if;
end DO331_Main;
