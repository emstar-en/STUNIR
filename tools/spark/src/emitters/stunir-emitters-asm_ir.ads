-- STUNIR Assembly IR Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: LLVM IR, GCC RTL, Various ISA IRs

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.ASM_IR is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type ASM_IR_Format is (LLVM_IR, GCC_RTL, MLIR, QBE_IR, Cranelift_IR);

   type ASM_IR_Config is record
      Format         : ASM_IR_Format := LLVM_IR;
      Optimize       : Boolean := False;
      Debug_Info     : Boolean := True;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant ASM_IR_Config :=
     (Format => LLVM_IR, Optimize => False, Debug_Info => True, Indent_Size => 2, Max_Line_Width => 100);

   type ASM_IR_Emitter is new Base_Emitter with record
      Config : ASM_IR_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out ASM_IR_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out ASM_IR_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out ASM_IR_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.ASM_IR;
