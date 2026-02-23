-- STUNIR Futhark Emitter (SPARK Body)
-- DO-178C Level A


with IR.Types; use IR.Types;

package body STUNIR.Emitters.Futhark_Family is
   pragma SPARK_Mode (On);

   procedure Emit_Module
     (Self   : in out Futhark_Emitter;
      Module : in     IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes);
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 2);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "-- STUNIR Futhark Emitter", Ok);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "-- Module: " & IR.Types.Name_Strings.To_String (Module.Module_Name), Ok);
      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   procedure Emit_Type
     (Self   : in out Futhark_Emitter;
      T      : in     IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes, T);
   begin
      Output := STUNIR.Emitters.CodeGen.Code_Buffers.Null_Bounded_String;
      Success := True;
   end Emit_Type;

   procedure Emit_Function
     (Self   : in out Futhark_Emitter;
      Func   : in     IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes);
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
      Name_Str : constant String := IR.Types.Name_Strings.To_String (Func.Base.Decl_Name);
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 2);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "let " & Name_Str & " =", Ok);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "  0", Ok);
      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Futhark_Family;
