-- STUNIR Lean4 Emitter (SPARK Body)
-- DO-178C Level A

with STUNIR.Emitters.CodeGen;

package body STUNIR.Emitters.Lean4_Family is
   pragma SPARK_Mode (On);

   procedure Emit_Module
     (Self   : in out Lean4_Emitter;
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes);
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 2);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "-- STUNIR Lean4 Emitter", Ok);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "namespace " & Semantic_IR.Types.Name_Strings.To_String (Module.Module_Name), Ok);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "", Ok);
      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   procedure Emit_Type
     (Self   : in out Lean4_Emitter;
      T      : in     Semantic_IR.Declarations.Type_Declaration;
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
     (Self   : in out Lean4_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes);
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
      Name_Str : constant String := Semantic_IR.Types.Name_Strings.To_String (Func.Base.Decl_Name);
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 2);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "def " & Name_Str & " :=" , Ok);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "  0", Ok);
      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Lean4_Family;
