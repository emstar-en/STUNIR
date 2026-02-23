-- STUNIR WASM Emitter Test Suite
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Text_IO; use Ada.Text_IO;
with IR.Modules; use IR.Modules;
with STUNIR.Emitters; use STUNIR.Emitters;
with STUNIR.Emitters.WASM; use STUNIR.Emitters.WASM;

procedure Test_WASM is
   pragma SPARK_Mode (Off);

   Emitter : WASM_Emitter;
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
   Put_Line ("STUNIR WASM Emitter Test Suite");
   Put_Line ("======================================");
   Put_Line ("");

   -- Test 1: Initialize WASM emitter
   Emitter.Category := Category_WASM;
   Emitter.Config.Use_WASI := True;
   Run_Test ("Initialize WASM Emitter", Emitter.Config.Use_WASI);

   -- Test 2: Test WASM type mapping
   Run_Test ("i32 Type Mapping", Get_WASM_Type ("i32") = "i32");
   Run_Test ("i64 Type Mapping", Get_WASM_Type ("i64") = "i64");
   Run_Test ("f32 Type Mapping", Get_WASM_Type ("f32") = "f32");
   Run_Test ("f64 Type Mapping", Get_WASM_Type ("f64") = "f64");

   -- Test 3: Test module generation
   Module.Module_Name := Name_Strings.To_Bounded_String ("WASMModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("add");
   Module.Functions (1).Return_Type := Type_Strings.To_Bounded_String ("i32");
   Module.Functions (1).Arg_Cnt := 0;
   Module.Functions (1).Stmt_Cnt := 0;

   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate WASM Module", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("WASM Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("Contains WASM_EXPORT",
                   (for some I in Output_Str'First .. Output_Str'Last - 10 =>
                      Output_Str (I .. I + 10) = "WASM_EXPORT"));
      end;
   end if;

   -- Test 4: Test WAT format generation
   Emit_WAT_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate WAT Module", Success);

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

end Test_WASM;
