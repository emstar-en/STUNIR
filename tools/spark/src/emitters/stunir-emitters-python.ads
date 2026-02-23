-- STUNIR Python Emitter (SPARK Specification)
-- DO-178C Level A

with STUNIR.Emitters;
with STUNIR.Emitters.Node_Table;
with IR.Modules;
with IR.Declarations;
with IR.Nodes;
with STUNIR.Emitters.CodeGen;

package STUNIR.Emitters.Python is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   type Python_Config is record
      Include_Types : Boolean := True;
      Include_Docs  : Boolean := True;
   end record;

   type Python_Emitter is new Base_Emitter with record
      Config : Python_Config;
   end record;

   procedure Emit_Module
     (Self   : in out Python_Emitter;
      Module : in     IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Type
     (Self   : in out Python_Emitter;
      T      : in     IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Emit_Function
     (Self   : in out Python_Emitter;
      Func   : in     IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.Python;
