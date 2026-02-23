-- STUNIR C Family Emitter (SPARK Specification)
-- DO-178C Level A

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with IR.Modules;
with IR.Declarations;
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

   procedure Emit_Module
     (Self   : in out C_Emitter;
      Module : in     IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Type
     (Self   : in out C_Emitter;
      T      : in     IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Function
     (Self   : in out C_Emitter;
      Func   : in     IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.CFamily;
