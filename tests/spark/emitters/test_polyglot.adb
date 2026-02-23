-- STUNIR Polyglot Emitter Test Suite
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Text_IO; use Ada.Text_IO;
with IR.Modules; use IR.Modules;
with STUNIR.Emitters; use STUNIR.Emitters;
with STUNIR.Emitters.Polyglot; use STUNIR.Emitters.Polyglot;

procedure Test_Polyglot is
   pragma SPARK_Mode (Off);

   Emitter : Polyglot_Emitter;
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
   Put_Line ("STUNIR Polyglot Emitter Test Suite");
   Put_Line ("======================================");
   Put_Line ("");

   -- Test 1: Initialize Polyglot emitter
   Emitter.Category := Category_Polyglot;
   Emitter.Config.Language := Lang_C99;
   Run_Test ("Initialize Polyglot Emitter", Emitter.Config.Language = Lang_C99);

   -- Test 2: Test language names
   Run_Test ("C89 Language Name", Get_Language_Name (Lang_C89) = "C89");
   Run_Test ("C99 Language Name", Get_Language_Name (Lang_C99) = "C99");
   Run_Test ("Rust Language Name", Get_Language_Name (Lang_Rust) = "Rust");

   -- Test 3: Test C type mapping
   Run_Test ("C i32 Type Mapping", Map_Type_To_C ("i32") = "int32_t");
   Run_Test ("C float Type Mapping", Map_Type_To_C ("f32") = "float");
   Run_Test ("C bool Type Mapping", Map_Type_To_C ("bool") = "bool");

   -- Test 4: Test Rust type mapping
   Run_Test ("Rust i32 Type Mapping", Map_Type_To_Rust ("i32") = "i32");
   Run_Test ("Rust f32 Type Mapping", Map_Type_To_Rust ("f32") = "f32");
   Run_Test ("Rust bool Type Mapping", Map_Type_To_Rust ("bool") = "bool");
   Run_Test ("Rust String Type Mapping", Map_Type_To_Rust ("string") = "String");

   -- Test 5: Test C89 generation
   Module.Module_Name := Name_Strings.To_Bounded_String ("TestModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("test");
   Module.Functions (1).Return_Type := Type_Strings.To_Bounded_String ("void");
   Module.Functions (1).Arg_Cnt := 0;
   Module.Functions (1).Stmt_Cnt := 0;

   Emit_C89 (Module, Output, Success);
   Run_Test ("Generate C89 Code", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("C89 Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("C89 Contains typedef",
                   (for some I in Output_Str'First .. Output_Str'Last - 6 =>
                      Output_Str (I .. I + 6) = "typedef"));
      end;
   end if;

   -- Test 6: Test C99 generation
   Emit_C99 (Module, Output, Success);
   Run_Test ("Generate C99 Code", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("C99 Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("C99 Contains stdint.h",
                   (for some I in Output_Str'First .. Output_Str'Last - 7 =>
                      Output_Str (I .. I + 7) = "stdint.h"));
      end;
   end if;

   -- Test 7: Test Rust generation
   Emit_Rust (Module, Output, Success);
   Run_Test ("Generate Rust Code", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Rust Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("Rust Contains pub fn",
                   (for some I in Output_Str'First .. Output_Str'Last - 5 =>
                      Output_Str (I .. I + 5) = "pub fn"));
      end;
   end if;

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

end Test_Polyglot;
