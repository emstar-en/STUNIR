--  STUNIR x86 Assembly Emitter - Ada SPARK Specification
--  Emit x86/x86_64 assembly code
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package X86_Emitter is

   type X86_Mode is (X86_32, X86_64);
   type X86_Syntax is (Intel_Syntax, AT_T_Syntax);

   type X86_Config is record
      Mode     : X86_Mode;
      Syntax   : X86_Syntax;
      Has_SSE  : Boolean;
      Has_AVX  : Boolean;
   end record;

   Default_Config : constant X86_Config := (
      Mode    => X86_64,
      Syntax  => AT_T_Syntax,
      Has_SSE => True,
      Has_AVX => False
   );

   procedure Emit_Prologue (
      Func_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in X86_Config;
      Status    : out Emitter_Status);

   procedure Emit_Epilogue (
      Content   : out Content_String;
      Config    : in X86_Config;
      Status    : out Emitter_Status);

end X86_Emitter;
