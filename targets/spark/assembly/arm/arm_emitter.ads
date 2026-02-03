--  STUNIR ARM Assembly Emitter - Ada SPARK Specification
--  Emit ARM/ARM64 assembly code
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package ARM_Emitter is

   type ARM_Mode is (ARM32, Thumb, ARM64_AArch64);
   type ARM_Variant is (Cortex_M0, Cortex_M3, Cortex_M4, Cortex_A53, Cortex_A72, Generic_ARM);

   type ARM_Config is record
      Mode       : ARM_Mode;
      Variant    : ARM_Variant;
      Has_FPU    : Boolean;
      Has_NEON   : Boolean;
   end record;

   Default_Config : constant ARM_Config := (
      Mode     => ARM32,
      Variant  => Cortex_M4,
      Has_FPU  => True,
      Has_NEON => False
   );

   procedure Emit_Prologue (
      Func_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status);

   procedure Emit_Epilogue (
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status);

   procedure Emit_Load (
      Reg       : in Natural;
      Offset    : in Integer;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status);

   procedure Emit_Store (
      Reg       : in Natural;
      Offset    : in Integer;
      Content   : out Content_String;
      Config    : in ARM_Config;
      Status    : out Emitter_Status);

end ARM_Emitter;
