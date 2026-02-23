-- STUNIR Lean4 Emitter (SPARK Specification)
-- DO-178C Level A

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with IR.Modules;
with IR.Declarations;
with STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Lean4_Family is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Lean4_Config is record
      Include_Docs : Boolean := True;
   end record;

   type Lean4_Emitter is new Base_Emitter with record
      Config : Lean4_Config;
   end record;

   procedure Emit_Module
     (Self   : in out Lean4_Emitter;
      Module : in     IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Type
     (Self   : in out Lean4_Emitter;
      T      : in     IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Function
     (Self   : in out Lean4_Emitter;
      Func   : in     IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.Lean4_Family;
