-- STUNIR Assembly Emitter Test Suite
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Text_IO; use Ada.Text_IO;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters; use STUNIR.Emitters;
with STUNIR.Emitters.Assembly; use STUNIR.Emitters.Assembly;

procedure Test_Assembly is
   pragma SPARK_Mode (Off);

   Emitter : Assembly_Emitter;
   Module  : IR_Module;
   Output  : IR_Code_Buffer;
   Success : Boolean;

   Test_Count  : Natural := 0;
   Pass_Count  : Natural := 0;

   procedure Run_Test (Test_Name : String; Condition : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Condition then
         Pass_Count := Pass_Count + 1;
         Put_Line ("[PASS] " & Test_Name);
      else
         Put_Line ("[FAIL] " & Test_Name);
      end if;
   end Run_Test;

begin
   Put_Line ("======================================");
   Put_Line ("STUNIR Assembly Emitter Test Suite");
   Put_Line ("======================================");
   Put_Line ("");

   -- Test 1: Initialize Assembly emitter
   Emitter.Category := Category_Assembly;
   Emitter.Config.Target := Target_X86_64;
   Emitter.Config.Syntax := Syntax_Intel;
   Run_Test ("Initialize Assembly Emitter", Emitter.Config.Target = Target_X86_64);

   -- Test 2: Test target names
   Run_Test ("x86 Target Name", Get_Target_Name (Target_X86) = "x86");
   Run_Test ("x86_64 Target Name", Get_Target_Name (Target_X86_64) = "x86_64");
   Run_Test ("ARM Target Name", Get_Target_Name (Target_ARM) = "ARM");
   Run_Test ("ARM64 Target Name", Get_Target_Name (Target_ARM64) = "ARM64");

   -- Test 3: Test syntax names
   Run_Test ("Intel Syntax Name", Get_Syntax_Name (Syntax_Intel) = "Intel");
   Run_Test ("AT&T Syntax Name", Get_Syntax_Name (Syntax_ATT) = "AT&T");
   Run_Test ("ARM Syntax Name", Get_Syntax_Name (Syntax_ARM) = "ARM");

   -- Test 4: Test module generation
   Module.Module_Name := Name_Strings.To_Bounded_String ("AsmModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("main");
   Module.Functions (1).Return_Type := Type_Strings.To_Bounded_String ("void");
   Module.Functions (1).Arg_Cnt := 0;
   Module.Functions (1).Stmt_Cnt := 0;

   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate x86_64 Assembly Module", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Assembly Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("Contains .intel_syntax",
                   (for some I in Output_Str'First .. Output_Str'Last - 12 =>
                      Output_Str (I .. I + 12) = ".intel_syntax"));
      end;
   end if;

   -- Test 5: Test ARM assembly generation
   Emitter.Config.Target := Target_ARM;
   Emitter.Config.Syntax := Syntax_ARM;
   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate ARM Assembly Module", Success);

   -- Test Summary
   Put_Line ("");
   Put_Line ("======================================");
   Put_Line ("Test Summary:");
   Put_Line ("  Total Tests: " & Natural'Image (Test_Count));
   Put_Line ("  Passed:      " & Natural'Image (Pass_Count));
   Put_Line ("  Failed:      " & Natural'Image (Test_Count - Pass_Count));
   Put_Line ("======================================");

   if Pass_Count = Test_Count then
      Put_Line ("ALL TESTS PASSED!");
   else
      Put_Line ("SOME TESTS FAILED!");
   end if;

end Test_Assembly;
