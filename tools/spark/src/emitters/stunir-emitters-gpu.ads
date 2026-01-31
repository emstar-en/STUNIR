-- STUNIR GPU Emitter
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
-- Supports: CUDA, OpenCL, Metal, ROCm, Vulkan

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.GPU is
   pragma SPARK_Mode (On);

   type GPU_Platform is
     (Platform_CUDA, Platform_OpenCL, Platform_Metal,
      Platform_ROCm, Platform_Vulkan);

   Max_Compute_Cap_Length : constant := 10;
   subtype Compute_Cap_String is String (1 .. Max_Compute_Cap_Length);

   type GPU_Config is record
      Platform       : GPU_Platform := Platform_CUDA;
      Compute_Cap    : Compute_Cap_String := "sm_75     ";  -- Padded to 10 chars
      Use_Shared_Mem : Boolean := True;
      Max_Threads    : Positive := 1024;
      Use_SIMD       : Boolean := True;
   end record
   with Dynamic_Predicate => Max_Threads > 0;

   type GPU_Emitter is new Base_Emitter with record
      Config : GPU_Config;
   end record;

   -- Override base emitter methods
   overriding procedure Emit_Module
     (Self    : in out GPU_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Type
     (Self    : in out GPU_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Function
     (Self    : in out GPU_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   -- GPU-specific methods
   procedure Emit_Kernel
     (Self    : in out GPU_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Func.Arg_Cnt >= 0,
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_Memory_Transfer
     (Self    : in out GPU_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean);

   -- Utility functions
   function Get_Platform_Name (Platform : GPU_Platform) return String
   with
     Global => null,
     Post => Get_Platform_Name'Result'Length > 0;

   function Get_Kernel_Prefix (Platform : GPU_Platform) return String
   with
     Global => null;

end STUNIR.Emitters.GPU;
