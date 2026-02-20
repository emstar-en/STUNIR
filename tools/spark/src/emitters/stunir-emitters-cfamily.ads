-- STUNIR C Family Emitter (SPARK Specification)
-- DO-178C Level A

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.CFamily is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type C_Dialect is (C89, C99, C11);

   type C_Config is record
      Dialect      : C_Dialect := C99;
      Include_Docs : Boolean := True;
   end record;

   type C_Emitter is new Base_Emitter with record
      Config : C_Config;
   end record;

   overriding procedure Emit_Module
     (Self   : in out C_Emitter;
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   overriding procedure Emit_Type
     (Self   : in out C_Emitter;
      T      : in     Semantic_IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   overriding procedure Emit_Function
     (Self   : in out C_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.CFamily;
