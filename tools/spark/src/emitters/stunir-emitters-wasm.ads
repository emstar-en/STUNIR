-- STUNIR WebAssembly Emitter
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
-- Supports: WASM, WASI, SIMD, Bulk Memory Ops

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.WASM is
   pragma SPARK_Mode (On);

   type WASM_Feature is (Feature_SIMD, Feature_Bulk_Memory, Feature_Threads);
   type Feature_Set is array (WASM_Feature) of Boolean;

   type WASM_Config is record
      Use_WASI    : Boolean := True;
      Features    : Feature_Set := (others => False);
      Export_All  : Boolean := True;
      Optimize    : Boolean := True;
   end record;

   type WASM_Emitter is new Base_Emitter with record
      Config : WASM_Config;
   end record;

   -- Override base emitter methods
   overriding procedure Emit_Module
     (Self    : in out WASM_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Type
     (Self    : in out WASM_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Function
     (Self    : in out WASM_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   -- WASM-specific methods
   procedure Emit_WAT_Module
     (Self    : in out WASM_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Is_Valid_Module (Module),
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   -- Utility functions
   function Get_WASM_Type (IR_Type : String) return String
   with
     Global => null;

end STUNIR.Emitters.WASM;
