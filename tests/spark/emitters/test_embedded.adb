-- STUNIR Embedded Emitter Test Suite
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Bounded;
with IR.Modules; use IR.Modules;
with STUNIR.Emitters; use STUNIR.Emitters;
with STUNIR.Emitters.Embedded; use STUNIR.Emitters.Embedded;

procedure Test_Embedded is
   pragma SPARK_Mode (Off);  -- Test code doesn't need SPARK

   Emitter : Embedded_Emitter;
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
   Put_Line ("STUNIR Embedded Emitter Test Suite");
   Put_Line ("======================================");
   Put_Line ("");

   -- Test 1: Initialize emitter
   Emitter.Category := Category_Embedded;
   Emitter.Config.Arch := Arch_ARM;
   Emitter.Config.Stack_Size := 4096;
   Run_Test ("Initialize Embedded Emitter", Emitter.Config.Stack_Size = 4096);

   -- Test 2: Test architecture configuration
   Run_Test ("ARM Architecture Selected", Get_Arch_Name (Emitter.Config.Arch) = "ARM");
   Run_Test ("ARM Toolchain Name", Get_Toolchain_Name (Emitter.Config.Arch) = "arm-none-eabi");

   -- Test 3: Test ARM64 configuration
   Emitter.Config.Arch := Arch_ARM64;
   Run_Test ("ARM64 Architecture Selected", Get_Arch_Name (Emitter.Config.Arch) = "ARM64");
   Run_Test ("ARM64 Toolchain Name", Get_Toolchain_Name (Emitter.Config.Arch) = "aarch64-none-elf");

   -- Test 4: Test RISC-V configuration
   Emitter.Config.Arch := Arch_RISCV;
   Run_Test ("RISC-V Architecture Selected", Get_Arch_Name (Emitter.Config.Arch) = "RISC-V");

   -- Test 5: Test startup code generation
   Emit_Startup_Code (Emitter, Output, Success);
   Run_Test ("Generate Startup Code", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Startup Code Non-Empty", Output_Str'Length > 0);
         Run_Test ("Startup Contains Reset_Handler",
                   (for some I in Output_Str'First .. Output_Str'Last - 12 =>
                      Output_Str (I .. I + 12) = "Reset_Handler"));
      end;
   end if;

   -- Test 6: Test linker script generation
   Emit_Linker_Script (Emitter, Output, Success);
   Run_Test ("Generate Linker Script", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Linker Script Non-Empty", Output_Str'Length > 0);
         Run_Test ("Linker Contains MEMORY",
                   (for some I in Output_Str'First .. Output_Str'Last - 5 =>
                      Output_Str (I .. I + 5) = "MEMORY"));
      end;
   end if;

   -- Test 7: Test simple module generation
   Module.Module_Name := Name_Strings.To_Bounded_String ("TestModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("test_func");
   Module.Functions (1).Return_Type := Type_Strings.To_Bounded_String ("void");
   Module.Functions (1).Arg_Cnt := 0;
   Module.Functions (1).Stmt_Cnt := 0;

   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate Simple Module", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Module Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("Module Contains STUNIR Comment",
                   (for some I in Output_Str'First .. Output_Str'Last - 5 =>
                      Output_Str (I .. I + 5) = "STUNIR"));
         Run_Test ("Module Contains Function Name",
                   (for some I in Output_Str'First .. Output_Str'Last - 8 =>
                      Output_Str (I .. I + 8) = "test_func"));
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

end Test_Embedded;
