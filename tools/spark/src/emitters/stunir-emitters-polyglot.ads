-- STUNIR Polyglot Emitter
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
-- Supports: C89, C99, Rust

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Polyglot is
   pragma SPARK_Mode (On);

   type Target_Language is (Lang_C89, Lang_C99, Lang_Rust);

   type Polyglot_Config is record
      Language       : Target_Language := Lang_C99;
      Use_StdLib     : Boolean := True;
      Generate_Tests : Boolean := False;
      Strict_Mode    : Boolean := True;
   end record;

   type Polyglot_Emitter is new Base_Emitter with record
      Config : Polyglot_Config;
   end record;

   -- Override base emitter methods
   overriding procedure Emit_Module
     (Self    : in out Polyglot_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Type
     (Self    : in out Polyglot_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Function
     (Self    : in out Polyglot_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   -- Language-specific emitters
   procedure Emit_C89
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Is_Valid_Module (Module),
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_C99
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Is_Valid_Module (Module),
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_Rust
     (Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   with
     Pre  => Is_Valid_Module (Module),
     Post => (if Success then Code_Buffers.Length (Output) > 0);

   -- Utility functions
   function Get_Language_Name (Lang : Target_Language) return String
   with
     Global => null,
     Post => Get_Language_Name'Result'Length > 0;

   function Map_Type_To_C (IR_Type : String) return String
   with
     Global => null;

   function Map_Type_To_Rust (IR_Type : String) return String
   with
     Global => null;

end STUNIR.Emitters.Polyglot;
