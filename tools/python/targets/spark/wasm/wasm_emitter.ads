--  STUNIR WASM Emitter - Ada SPARK Specification
--  Emit WebAssembly text format (WAT) and binary
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package WASM_Emitter is

   type WASM_Config is record
      Enable_SIMD     : Boolean;
      Enable_Threads  : Boolean;
      Stack_Size      : Positive;
      Memory_Pages    : Positive;
   end record;

   Default_Config : constant WASM_Config := (
      Enable_SIMD    => False,
      Enable_Threads => False,
      Stack_Size     => 65536,
      Memory_Pages   => 1
   );

   type WASM_Type is (I32, I64, F32, F64, V128, FuncRef, ExternRef);

   procedure Emit_Module (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in WASM_Config;
      Status    : out Emitter_Status)
      with Pre => Identifier_Strings.Length (Name) > 0;

   procedure Emit_Function (
      Name      : in Identifier_String;
      Params    : in String;
      Results   : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status);

   function Map_Type_To_WASM (IR_Type : IR_Data_Type) return WASM_Type;

end WASM_Emitter;
