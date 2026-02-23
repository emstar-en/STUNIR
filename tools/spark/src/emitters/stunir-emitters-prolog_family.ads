-- STUNIR Prolog Emitter (SPARK Specification)
-- DO-178C Level A
--
-- DEPRECATED: This typed Semantic IR emitter is deprecated.
-- The canonical emitter path is now the Micro IR emitter in emit_target.adb.
-- All Prolog family targets (SWI-Prolog, GNU-Prolog, Mercury, generic Prolog)
-- are now supported via the unified pipeline.
-- See: src/emitters/emit_target.adb and src/emitters/emit_target_main.adb
-- Scheduled removal: 2026-06-01

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with IR.Modules;
with IR.Declarations;
with STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Prolog_Family is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Prolog_Config is record
      Include_Docs : Boolean := True;
   end record;

   type Prolog_Emitter is new Base_Emitter with record
      Config : Prolog_Config;
   end record;

   procedure Emit_Module
     (Self   : in out Prolog_Emitter;
      Module : in     IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Type
     (Self   : in out Prolog_Emitter;
      T      : in     IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Function
     (Self   : in out Prolog_Emitter;
      Func   : in     IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.Prolog_Family;
