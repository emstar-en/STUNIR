-- STUNIR Lisp Emitter Test Suite
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

with Ada.Text_IO; use Ada.Text_IO;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.Lisp; use STUNIR.Emitters.Lisp;

procedure Test_Lisp is
   pragma SPARK_Mode (Off);  -- Test code doesn't need SPARK mode

   Test_Count : Natural := 0;
   Pass_Count : Natural := 0;
   Fail_Count : Natural := 0;

   procedure Run_Test (Test_Name : String; Success : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Success then
         Put_Line ("[PASS] " & Test_Name);
         Pass_Count := Pass_Count + 1;
      else
         Put_Line ("[FAIL] " & Test_Name);
         Fail_Count := Fail_Count + 1;
      end if;
   end Run_Test;

   -- Test 1: Empty Module - Common Lisp
   procedure Test_Empty_Module_Common_Lisp is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("test_module");
      Module.Docstring   := Doc_Strings.To_Bounded_String ("Test module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Common_Lisp;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-001: Empty Module - Common Lisp",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Empty_Module_Common_Lisp;

   -- Test 2: Function Emission - Common Lisp
   procedure Test_Function_Common_Lisp is
      Emitter : Lisp_Emitter;
      Func    : IR_Function;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Func.Name := Name_Strings.To_Bounded_String ("add");
      Func.Docstring := Doc_Strings.To_Bounded_String ("Add two numbers");
      Func.Args (1).Name := Name_Strings.To_Bounded_String ("x");
      Func.Args (1).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Args (2).Name := Name_Strings.To_Bounded_String ("y");
      Func.Args (2).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Return_Type := Type_Strings.To_Bounded_String ("integer");
      Func.Arg_Cnt := 2;
      Func.Stmt_Cnt := 0;

      Emitter.Config.Dialect := Common_Lisp;
      Emitter.Emit_Function (Func, Output, Success);

      Run_Test ("TC-002: Function - Common Lisp",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Function_Common_Lisp;

   -- Test 3: Scheme R7RS
   procedure Test_Module_Scheme is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("math_ops");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Scheme;
      Emitter.Config.Scheme_Std := R7RS;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-003: Module - Scheme R7RS",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Scheme;

   -- Test 4: Clojure Namespace
   procedure Test_Module_Clojure is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("my.namespace");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Clojure;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-004: Namespace - Clojure",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Clojure;

   -- Test 5: Racket Module
   procedure Test_Module_Racket is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("racket_test");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Racket;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-005: Module - Racket",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Racket;

   -- Test 6: Emacs Lisp Module
   procedure Test_Module_Emacs_Lisp is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("my-package");
      Module.Docstring := Doc_Strings.To_Bounded_String ("My Emacs package");
      Module.Func_Cnt := 0;
      Module.Type_Cnt := 0;

      Emitter.Config.Dialect := Emacs_Lisp;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-006: Module - Emacs Lisp",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Emacs_Lisp;

   -- Test 7: Guile Module
   procedure Test_Module_Guile is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("guile_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Guile;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-007: Module - Guile",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Guile;

   -- Test 8: Hy Module
   procedure Test_Module_Hy is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("hy_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Hy;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-008: Module - Hy",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Hy;

   -- Test 9: Janet Module
   procedure Test_Module_Janet is
      Emitter : Lisp_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("janet_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := Janet;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-009: Module - Janet",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Janet;

   -- Test 10: Type Definition - Clojure
   procedure Test_Type_Clojure is
      Emitter : Lisp_Emitter;
      T       : IR_Type_Def;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      T.Name := Name_Strings.To_Bounded_String ("Person");
      T.Docstring := Doc_Strings.To_Bounded_String ("Person record");
      T.Fields (1).Name := Name_Strings.To_Bounded_String ("name");
      T.Fields (1).Type_Ref := Type_Strings.To_Bounded_String ("string");
      T.Fields (2).Name := Name_Strings.To_Bounded_String ("age");
      T.Fields (2).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      T.Field_Cnt := 2;

      Emitter.Config.Dialect := Clojure;
      Emitter.Emit_Type (T, Output, Success);

      Run_Test ("TC-010: Type - Clojure",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Type_Clojure;

   -- Test 11: Deterministic Output
   procedure Test_Deterministic_Output is
      Emitter1 : Lisp_Emitter;
      Emitter2 : Lisp_Emitter;
      Module   : IR_Module;
      Output1  : IR_Code_Buffer;
      Output2  : IR_Code_Buffer;
      Success1 : Boolean;
      Success2 : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("determinism_test");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter1.Config.Dialect := Common_Lisp;
      Emitter2.Config.Dialect := Common_Lisp;

      Emitter1.Emit_Module (Module, Output1, Success1);
      Emitter2.Emit_Module (Module, Output2, Success2);

      declare
         Match : constant Boolean :=
           Success1 and Success2 and
           Code_Buffers.To_String (Output1) = Code_Buffers.To_String (Output2);
      begin
         Run_Test ("TC-011: Deterministic Output", Match);
      end;
   end Test_Deterministic_Output;

begin
   Put_Line ("=================================================");
   Put_Line ("STUNIR Lisp Emitter Test Suite - Phase 3b");
   Put_Line ("DO-178C Level A Compliance Testing");
   Put_Line ("=================================================");
   New_Line;

   -- Run all tests
   Test_Empty_Module_Common_Lisp;
   New_Line;

   Test_Function_Common_Lisp;
   New_Line;

   Test_Module_Scheme;
   New_Line;

   Test_Module_Clojure;
   New_Line;

   Test_Module_Racket;
   New_Line;

   Test_Module_Emacs_Lisp;
   New_Line;

   Test_Module_Guile;
   New_Line;

   Test_Module_Hy;
   New_Line;

   Test_Module_Janet;
   New_Line;

   Test_Type_Clojure;
   New_Line;

   Test_Deterministic_Output;
   New_Line;

   -- Summary
   Put_Line ("=================================================");
   Put_Line ("Test Summary:");
   Put_Line ("  Total Tests: " & Natural'Image (Test_Count));
   Put_Line ("  Passed:      " & Natural'Image (Pass_Count));
   Put_Line ("  Failed:      " & Natural'Image (Fail_Count));
   Put_Line ("=================================================");

   if Fail_Count = 0 then
      Put_Line ("✅ ALL TESTS PASSED");
   else
      Put_Line ("❌ SOME TESTS FAILED");
   end if;
end Test_Lisp;
