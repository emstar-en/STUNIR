-- STUNIR Assembly Emitter
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
-- Supports: x86, ARM assembly

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Assembly is
   pragma SPARK_Mode (On);

   type Assembly_Target is (Target_X86, Target_X86_64, Target_ARM, Target_ARM64);
   type Assembly_Syntax is (Syntax_Intel, Syntax_ATT, Syntax_ARM);

   type Assembly_Config is record
      Target     : Assembly_Target := Target_X86_64;
      Syntax     : Assembly_Syntax := Syntax_Intel;
      Optimize   : Boolean := True;
      Add_Debug  : Boolean := False;
   end record;

   type Assembly_Emitter is new Base_Emitter with record
      Config : Assembly_Config;
   end record;

   -- Override base emitter methods
   overriding procedure Emit_Module
     (Self    : in out Assembly_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Type
     (Self    : in out Assembly_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   overriding procedure Emit_Function
     (Self    : in out Assembly_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean);

   -- Assembly-specific methods
   procedure Emit_Function_Prologue
     (Self    : in out Assembly_Emitter;
      Gen     : in out Code_Generator;
      Func    : in     IR_Function;
      Success :    out Boolean)
   with
     Pre => Func.Arg_Cnt >= 0;

   procedure Emit_Function_Epilogue
     (Self    : in out Assembly_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean);

   -- Utility functions
   function Get_Target_Name (Target : Assembly_Target) return String
   with
     Global => null,
     Post => Get_Target_Name'Result'Length > 0;

   function Get_Syntax_Name (Syntax : Assembly_Syntax) return String
   with
     Global => null,
     Post => Get_Syntax_Name'Result'Length > 0;

end STUNIR.Emitters.Assembly;
