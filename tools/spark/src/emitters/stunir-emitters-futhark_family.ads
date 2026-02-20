-- STUNIR Futhark Emitter (SPARK Specification)
-- DO-178C Level A

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Futhark_Family is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Futhark_Config is record
      Include_Docs : Boolean := True;
   end record;

   type Futhark_Emitter is new Base_Emitter with record
      Config : Futhark_Config;
   end record;

   overriding procedure Emit_Module
     (Self   : in out Futhark_Emitter;
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   overriding procedure Emit_Type
     (Self   : in out Futhark_Emitter;
      T      : in     Semantic_IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   overriding procedure Emit_Function
     (Self   : in out Futhark_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.Futhark_Family;
