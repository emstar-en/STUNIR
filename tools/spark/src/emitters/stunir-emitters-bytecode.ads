-- STUNIR Bytecode Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: JVM Bytecode, .NET IL, Python Bytecode

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Bytecode is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Bytecode_Format is (JVM_Bytecode, DOTNET_IL, Python_Bytecode, LLVM_IR, WebAssembly_Bytecode);

   type Bytecode_Config is record
      Format         : Bytecode_Format := JVM_Bytecode;
      Stack_Size     : Positive := 256;
      Optimize_Level : Natural := 0;
      Debug_Symbols  : Boolean := True;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   Default_Config : constant Bytecode_Config :=
     (Format => JVM_Bytecode, Stack_Size => 256, Optimize_Level => 0, Debug_Symbols => True, Indent_Size => 2, Max_Line_Width => 100);

   type Bytecode_Emitter is new Base_Emitter with record
      Config : Bytecode_Config := Default_Config;
   end record;

   overriding procedure Emit_Module (Self : in out Bytecode_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Is_Valid_Module (Module), Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type (Self : in out Bytecode_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => T.Field_Cnt > 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function (Self : in out Bytecode_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean)
   with Pre'Class => Func.Arg_Cnt >= 0, Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Bytecode;
