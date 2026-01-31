-- STUNIR Prolog Emitter Test Suite
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

with Ada.Text_IO; use Ada.Text_IO;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.Prolog; use STUNIR.Emitters.Prolog;

procedure Test_Prolog is
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

   -- Test P001: Empty Module - SWI-Prolog
   procedure Test_Empty_Module_SWI is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("test_module");
      Module.Docstring   := Doc_Strings.To_Bounded_String ("Test module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := SWI_Prolog;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P001: Empty Module - SWI-Prolog",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Empty_Module_SWI;

   -- Test P002: Module with CLP - SWI-Prolog
   procedure Test_CLP_SWI is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("clp_test");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := SWI_Prolog;
      Emitter.Config.Use_CLP := True;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P002: Module with CLP - SWI-Prolog",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_CLP_SWI;

   -- Test P003: GNU Prolog Module
   procedure Test_Module_GNU is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("gnu_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := GNU_Prolog;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P003: Module - GNU Prolog",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_GNU;

   -- Test P004: SICStus Prolog Module
   procedure Test_Module_SICStus is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("sicstus_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := SICStus;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P004: Module - SICStus",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_SICStus;

   -- Test P005: YAP Module with Tabling
   procedure Test_Module_YAP_Tabling is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Func    : IR_Function;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      -- Create function
      Func.Name := Name_Strings.To_Bounded_String ("fibonacci");
      Func.Docstring := Doc_Strings.To_Bounded_String ("Fibonacci with tabling");
      Func.Args (1).Name := Name_Strings.To_Bounded_String ("n");
      Func.Args (1).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Return_Type := Type_Strings.To_Bounded_String ("integer");
      Func.Arg_Cnt := 1;
      Func.Stmt_Cnt := 0;

      -- Create module
      Module.Module_Name := Name_Strings.To_Bounded_String ("yap_module");
      Module.Functions (1) := Func;
      Module.Func_Cnt := 1;
      Module.Type_Cnt := 0;

      Emitter.Config.Dialect := YAP;
      Emitter.Config.Use_Tabling := True;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P005: YAP with Tabling",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_YAP_Tabling;

   -- Test P006: XSB Module with Tabling
   procedure Test_Module_XSB is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("xsb_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := XSB;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P006: Module - XSB",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_XSB;

   -- Test P007: Ciao Prolog with Assertions
   procedure Test_Module_Ciao_Assertions is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Func    : IR_Function;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      -- Create function
      Func.Name := Name_Strings.To_Bounded_String ("add");
      Func.Docstring := Doc_Strings.To_Bounded_String ("Add two integers");
      Func.Args (1).Name := Name_Strings.To_Bounded_String ("x");
      Func.Args (1).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Args (2).Name := Name_Strings.To_Bounded_String ("y");
      Func.Args (2).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Return_Type := Type_Strings.To_Bounded_String ("integer");
      Func.Arg_Cnt := 2;
      Func.Stmt_Cnt := 0;

      -- Create module
      Module.Module_Name := Name_Strings.To_Bounded_String ("ciao_module");
      Module.Functions (1) := Func;
      Module.Func_Cnt := 1;
      Module.Type_Cnt := 0;

      Emitter.Config.Dialect := Ciao;
      Emitter.Config.Use_Assertions := True;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P007: Ciao with Assertions",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_Ciao_Assertions;

   -- Test P008: B-Prolog Module
   procedure Test_Module_BProlog is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("bprolog_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := BProlog;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P008: Module - B-Prolog",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_BProlog;

   -- Test P009: ECLiPSe Module with CLP
   procedure Test_Module_ECLiPSe_CLP is
      Emitter : Prolog_Emitter;
      Module  : IR_Module;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("eclipse_module");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter.Config.Dialect := ECLiPSe;
      Emitter.Config.Use_CLP := True;
      Emitter.Emit_Module (Module, Output, Success);

      Run_Test ("TC-P009: ECLiPSe with CLP",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Module_ECLiPSe_CLP;

   -- Test P010: Function to Predicate Conversion
   procedure Test_Function_To_Predicate is
      Emitter : Prolog_Emitter;
      Func    : IR_Function;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      Func.Name := Name_Strings.To_Bounded_String ("multiply");
      Func.Docstring := Doc_Strings.To_Bounded_String ("Multiply two numbers");
      Func.Args (1).Name := Name_Strings.To_Bounded_String ("a");
      Func.Args (1).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Args (2).Name := Name_Strings.To_Bounded_String ("b");
      Func.Args (2).Type_Ref := Type_Strings.To_Bounded_String ("integer");
      Func.Return_Type := Type_Strings.To_Bounded_String ("integer");
      Func.Arg_Cnt := 2;
      Func.Stmt_Cnt := 0;

      Emitter.Config.Dialect := SWI_Prolog;
      Emitter.Emit_Function (Func, Output, Success);

      Run_Test ("TC-P010: Function to Predicate",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Function_To_Predicate;

   -- Test P011: Type Definition
   procedure Test_Type_Definition is
      Emitter : Prolog_Emitter;
      T       : IR_Type_Def;
      Output  : IR_Code_Buffer;
      Success : Boolean;
   begin
      T.Name := Name_Strings.To_Bounded_String ("point");
      T.Docstring := Doc_Strings.To_Bounded_String ("2D Point");
      T.Fields (1).Name := Name_Strings.To_Bounded_String ("x");
      T.Fields (1).Type_Ref := Type_Strings.To_Bounded_String ("float");
      T.Fields (2).Name := Name_Strings.To_Bounded_String ("y");
      T.Fields (2).Type_Ref := Type_Strings.To_Bounded_String ("float");
      T.Field_Cnt := 2;

      Emitter.Config.Dialect := Ciao;
      Emitter.Emit_Type (T, Output, Success);

      Run_Test ("TC-P011: Type Definition",
                Success and Code_Buffers.Length (Output) > 0);
      if Success then
         Put_Line ("  Generated Code:");
         Put_Line (Code_Buffers.To_String (Output));
      end if;
   end Test_Type_Definition;

   -- Test P012: Dialect Feature Support
   procedure Test_Dialect_Features is
      Tabling_YAP : constant Boolean := Supports_Tabling (YAP);
      Tabling_XSB : constant Boolean := Supports_Tabling (XSB);
      Tabling_SWI : constant Boolean := Supports_Tabling (SWI_Prolog);
      CLP_SWI     : constant Boolean := Supports_CLP (SWI_Prolog);
      CLP_ECLiPSe : constant Boolean := Supports_CLP (ECLiPSe);
      Assert_Ciao : constant Boolean := Supports_Assertions (Ciao);
   begin
      Run_Test ("TC-P012: Dialect Features",
                Tabling_YAP and Tabling_XSB and not Tabling_SWI and
                CLP_SWI and CLP_ECLiPSe and Assert_Ciao);
      Put_Line ("  YAP Tabling: " & Boolean'Image (Tabling_YAP));
      Put_Line ("  XSB Tabling: " & Boolean'Image (Tabling_XSB));
      Put_Line ("  SWI CLP:     " & Boolean'Image (CLP_SWI));
      Put_Line ("  Ciao Assertions: " & Boolean'Image (Assert_Ciao));
   end Test_Dialect_Features;

   -- Test P013: Deterministic Output
   procedure Test_Deterministic_Output is
      Emitter1 : Prolog_Emitter;
      Emitter2 : Prolog_Emitter;
      Module   : IR_Module;
      Output1  : IR_Code_Buffer;
      Output2  : IR_Code_Buffer;
      Success1 : Boolean;
      Success2 : Boolean;
   begin
      Module.Module_Name := Name_Strings.To_Bounded_String ("determinism_test");
      Module.Func_Cnt    := 0;
      Module.Type_Cnt    := 0;

      Emitter1.Config.Dialect := SWI_Prolog;
      Emitter2.Config.Dialect := SWI_Prolog;

      Emitter1.Emit_Module (Module, Output1, Success1);
      Emitter2.Emit_Module (Module, Output2, Success2);

      declare
         Match : constant Boolean :=
           Success1 and Success2 and
           Code_Buffers.To_String (Output1) = Code_Buffers.To_String (Output2);
      begin
         Run_Test ("TC-P013: Deterministic Output", Match);
      end;
   end Test_Deterministic_Output;

begin
   Put_Line ("=================================================");
   Put_Line ("STUNIR Prolog Emitter Test Suite - Phase 3b");
   Put_Line ("DO-178C Level A Compliance Testing");
   Put_Line ("=================================================");
   New_Line;

   -- Run all tests
   Test_Empty_Module_SWI;
   New_Line;

   Test_CLP_SWI;
   New_Line;

   Test_Module_GNU;
   New_Line;

   Test_Module_SICStus;
   New_Line;

   Test_Module_YAP_Tabling;
   New_Line;

   Test_Module_XSB;
   New_Line;

   Test_Module_Ciao_Assertions;
   New_Line;

   Test_Module_BProlog;
   New_Line;

   Test_Module_ECLiPSe_CLP;
   New_Line;

   Test_Function_To_Predicate;
   New_Line;

   Test_Type_Definition;
   New_Line;

   Test_Dialect_Features;
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
end Test_Prolog;
