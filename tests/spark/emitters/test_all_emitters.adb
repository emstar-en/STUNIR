-- STUNIR Comprehensive Emitter Test Suite
-- DO-178C Level A
-- Phase 3c: Testing all 17 remaining category emitters

with Ada.Text_IO; use Ada.Text_IO;
with IR.Modules; use IR.Modules;
with STUNIR.Emitters;
with STUNIR.Emitters.Business;
with STUNIR.Emitters.FPGA;
with STUNIR.Emitters.Grammar;
with STUNIR.Emitters.Lexer;
with STUNIR.Emitters.Parser;
with STUNIR.Emitters.Expert;
with STUNIR.Emitters.Constraints;
with STUNIR.Emitters.Functional;
with STUNIR.Emitters.OOP;
with STUNIR.Emitters.Mobile;
with STUNIR.Emitters.Scientific;
with STUNIR.Emitters.Bytecode;
with STUNIR.Emitters.Systems;
with STUNIR.Emitters.Planning;
with STUNIR.Emitters.ASM_IR;
with STUNIR.Emitters.BEAM;
with STUNIR.Emitters.ASP;

procedure Test_All_Emitters is
   Module  : IR_Module;
   Output  : IR_Code_Buffer;
   Success : Boolean;
   Test_Count : Natural := 0;
   Pass_Count : Natural := 0;

   procedure Test_Emitter (Name : String; Test_Passed : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Test_Passed then
         Pass_Count := Pass_Count + 1;
         Put_Line ("[PASS] " & Name);
      else
         Put_Line ("[FAIL] " & Name);
      end if;
   end Test_Emitter;

begin
   Put_Line ("STUNIR Phase 3c: Comprehensive Emitter Test Suite");
   Put_Line ("==================================================");
   Put_Line ("");

   -- Initialize test module
   Module.Module_Name := Name_Strings.To_Bounded_String ("TestModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("test_func");
   Module.Functions (1).Arg_Cnt := 0;

   -- Test 1: Business Emitter (COBOL/BASIC)
   declare
      Emitter : STUNIR.Emitters.Business.Business_Emitter;
   begin
      STUNIR.Emitters.Business.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Business Emitter (COBOL/BASIC)", Success);
   end;

   -- Test 2: FPGA Emitter (VHDL/Verilog)
   declare
      Emitter : STUNIR.Emitters.FPGA.FPGA_Emitter;
   begin
      STUNIR.Emitters.FPGA.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("FPGA Emitter (VHDL/Verilog)", Success);
   end;

   -- Test 3: Grammar Emitter
   declare
      Emitter : STUNIR.Emitters.Grammar.Grammar_Emitter;
   begin
      STUNIR.Emitters.Grammar.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Grammar Emitter (ANTLR/PEG/BNF)", Success);
   end;

   -- Test 4: Lexer Emitter
   declare
      Emitter : STUNIR.Emitters.Lexer.Lexer_Emitter;
   begin
      STUNIR.Emitters.Lexer.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Lexer Emitter (Flex/JFlex)", Success);
   end;

   -- Test 5: Parser Emitter
   declare
      Emitter : STUNIR.Emitters.Parser.Parser_Emitter;
   begin
      STUNIR.Emitters.Parser.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Parser Emitter (Yacc/ANTLR)", Success);
   end;

   -- Test 6: Expert Systems Emitter
   declare
      Emitter : STUNIR.Emitters.Expert.Expert_Emitter;
   begin
      STUNIR.Emitters.Expert.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Expert Systems Emitter (CLIPS/Jess)", Success);
   end;

   -- Test 7: Constraints Emitter
   declare
      Emitter : STUNIR.Emitters.Constraints.Constraints_Emitter;
   begin
      STUNIR.Emitters.Constraints.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Constraints Emitter (MiniZinc/Z3)", Success);
   end;

   -- Test 8: Functional Emitter
   declare
      Emitter : STUNIR.Emitters.Functional.Functional_Emitter;
   begin
      STUNIR.Emitters.Functional.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Functional Emitter (Haskell/OCaml)", Success);
   end;

   -- Test 9: OOP Emitter
   declare
      Emitter : STUNIR.Emitters.OOP.OOP_Emitter;
   begin
      STUNIR.Emitters.OOP.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("OOP Emitter (Java/C++/C#)", Success);
   end;

   -- Test 10: Mobile Emitter
   declare
      Emitter : STUNIR.Emitters.Mobile.Mobile_Emitter;
   begin
      STUNIR.Emitters.Mobile.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Mobile Emitter (Swift/Kotlin)", Success);
   end;

   -- Test 11: Scientific Emitter
   declare
      Emitter : STUNIR.Emitters.Scientific.Scientific_Emitter;
   begin
      STUNIR.Emitters.Scientific.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Scientific Emitter (MATLAB/NumPy)", Success);
   end;

   -- Test 12: Bytecode Emitter
   declare
      Emitter : STUNIR.Emitters.Bytecode.Bytecode_Emitter;
   begin
      STUNIR.Emitters.Bytecode.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Bytecode Emitter (JVM/.NET)", Success);
   end;

   -- Test 13: Systems Emitter
   declare
      Emitter : STUNIR.Emitters.Systems.Systems_Emitter;
   begin
      STUNIR.Emitters.Systems.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Systems Emitter (Ada/D/Zig)", Success);
   end;

   -- Test 14: Planning Emitter
   declare
      Emitter : STUNIR.Emitters.Planning.Planning_Emitter;
   begin
      STUNIR.Emitters.Planning.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("Planning Emitter (PDDL)", Success);
   end;

   -- Test 15: ASM IR Emitter
   declare
      Emitter : STUNIR.Emitters.ASM_IR.ASM_IR_Emitter;
   begin
      STUNIR.Emitters.ASM_IR.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("ASM IR Emitter (LLVM IR)", Success);
   end;

   -- Test 16: BEAM Emitter
   declare
      Emitter : STUNIR.Emitters.BEAM.BEAM_Emitter;
   begin
      STUNIR.Emitters.BEAM.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("BEAM Emitter (Erlang/Elixir)", Success);
   end;

   -- Test 17: ASP Emitter
   declare
      Emitter : STUNIR.Emitters.ASP.ASP_Emitter;
   begin
      STUNIR.Emitters.ASP.Emit_Module (Emitter, Module, Output, Success);
      Test_Emitter ("ASP Emitter (Clingo)", Success);
   end;

   Put_Line ("");
   Put_Line ("==================================================");
   Put_Line ("Test Summary:");
   Put_Line ("  Total Tests: " & Natural'Image (Test_Count));
   Put_Line ("  Passed:      " & Natural'Image (Pass_Count));
   Put_Line ("  Failed:      " & Natural'Image (Test_Count - Pass_Count));
   
   if Pass_Count = Test_Count then
      Put_Line ("");
      Put_Line ("✓ ALL TESTS PASSED!");
      Put_Line ("✓ Phase 3c: 17 Emitters Verified");
   else
      Put_Line ("");
      Put_Line ("✗ SOME TESTS FAILED");
   end if;

end Test_All_Emitters;
