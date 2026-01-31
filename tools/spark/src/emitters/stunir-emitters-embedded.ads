-- STUNIR Embedded Systems Emitter
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
-- Supports: ARM, ARM64, RISC-V, MIPS, AVR, x86

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Embedded is
   pragma SPARK_Mode (On);

   type Architecture is
     (Arch_ARM, Arch_ARM64, Arch_RISCV, Arch_MIPS, Arch_AVR, Arch_X86);

   type Embedded_Config is record
      Arch            : Architecture := Arch_ARM;
      Use_FPU         : Boolean := False;
      Stack_Size      : Positive := 4096;
      Heap_Size       : Positive := 2048;
      Generate_Linker : Boolean := True;
      Flash_Start     : Natural := 16#08000000#;
      Flash_Size      : Positive := 262144;  -- 256KB
      RAM_Start       : Natural := 16#20000000#;
      RAM_Size        : Positive := 65536;   -- 64KB
   end record
   with Dynamic_Predicate =>
     Stack_Size > 0 and Heap_Size > 0 and
     Flash_Size > 0 and RAM_Size > 0;

   type Embedded_Emitter is new Base_Emitter with record
      Config : Embedded_Config;
   end record;

   -- Override base emitter methods
   overriding procedure Emit_Module
     (Self    : in out Embedded_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Type
     (Self    : in out Embedded_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Function
     (Self    : in out Embedded_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   -- Embedded-specific methods
   procedure Emit_Startup_Code
     (Self    : in out Embedded_Emitter;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Self.Config.Stack_Size > 0,
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_Linker_Script
     (Self    : in out Embedded_Emitter;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Self.Config.Generate_Linker,
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_Memory_Config
     (Self    : in out Embedded_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean)
   with
     Pre  => Self.Config.RAM_Size > 0;

   -- Utility functions
   function Get_Arch_Name (Arch : Architecture) return String
   with
     Global => null,
     Post => Get_Arch_Name'Result'Length > 0;

   function Get_Toolchain_Name (Arch : Architecture) return String
   with
     Global => null,
     Post => Get_Toolchain_Name'Result'Length > 0;

end STUNIR.Emitters.Embedded;
