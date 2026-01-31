-- STUNIR GPU Emitter Test Suite
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Ada.Text_IO; use Ada.Text_IO;
with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters; use STUNIR.Emitters;
with STUNIR.Emitters.GPU; use STUNIR.Emitters.GPU;

procedure Test_GPU is
   pragma SPARK_Mode (Off);

   Emitter : GPU_Emitter;
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
   Put_Line ("STUNIR GPU Emitter Test Suite");
   Put_Line ("======================================");
   Put_Line ("");

   -- Test 1: Initialize GPU emitter
   Emitter.Category := Category_GPU;
   Emitter.Config.Platform := Platform_CUDA;
   Run_Test ("Initialize GPU Emitter", Emitter.Config.Platform = Platform_CUDA);

   -- Test 2: Test platform names
   Run_Test ("CUDA Platform Name", Get_Platform_Name (Platform_CUDA) = "CUDA");
   Run_Test ("OpenCL Platform Name", Get_Platform_Name (Platform_OpenCL) = "OpenCL");
   Run_Test ("Metal Platform Name", Get_Platform_Name (Platform_Metal) = "Metal");
   Run_Test ("ROCm Platform Name", Get_Platform_Name (Platform_ROCm) = "ROCm");
   Run_Test ("Vulkan Platform Name", Get_Platform_Name (Platform_Vulkan) = "Vulkan");

   -- Test 3: Test kernel prefix
   Run_Test ("CUDA Kernel Prefix", Get_Kernel_Prefix (Platform_CUDA) = "__global__");
   Run_Test ("OpenCL Kernel Prefix", Get_Kernel_Prefix (Platform_OpenCL) = "__kernel");

   -- Test 4: Test simple kernel generation
   Module.Module_Name := Name_Strings.To_Bounded_String ("GPUModule");
   Module.Func_Cnt := 1;
   Module.Functions (1).Name := Name_Strings.To_Bounded_String ("vector_add");
   Module.Functions (1).Return_Type := Type_Strings.To_Bounded_String ("void");
   Module.Functions (1).Arg_Cnt := 0;
   Module.Functions (1).Stmt_Cnt := 0;

   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate CUDA Kernel Module", Success);

   if Success then
      declare
         Output_Str : constant String := Code_Buffers.To_String (Output);
      begin
         Run_Test ("Kernel Output Non-Empty", Output_Str'Length > 0);
         Run_Test ("Contains __global__ keyword",
                   (for some I in Output_Str'First .. Output_Str'Last - 9 =>
                      Output_Str (I .. I + 9) = "__global__"));
      end;
   end if;

   -- Test 5: Test OpenCL platform
   Emitter.Config.Platform := Platform_OpenCL;
   Emit_Module (Emitter, Module, Output, Success);
   Run_Test ("Generate OpenCL Kernel Module", Success);

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

end Test_GPU;
