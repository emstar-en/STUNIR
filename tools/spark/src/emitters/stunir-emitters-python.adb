-- STUNIR Python Emitter (SPARK Body)
-- DO-178C Level A


with Semantic_IR.Types; use Semantic_IR.Types;
with STUNIR.Emitters.AST_Render;

package body STUNIR.Emitters.Python is
   pragma SPARK_Mode (On);

   procedure Emit_Module
     (Self   : in out Python_Emitter;
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 4);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "# STUNIR Python Emitter", Ok);
      if not Ok then
         Output := STUNIR.Emitters.CodeGen.Code_Buffers.Null_Bounded_String;
         Success := False;
         return;
      end if;
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "", Ok);

      -- Emit function stubs based on declarations list
      for I in 1 .. Module.Decl_Count loop
         declare
            Decl_Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
              STUNIR.Emitters.Node_Table.Lookup (Nodes, Module.Declarations (I));
         begin
            if Decl_Index > 0 then
               declare
                  Decl : constant STUNIR.Emitters.Node_Table.Declaration_Record :=
                    STUNIR.Emitters.Node_Table.Get_Declaration (Nodes, Decl_Index);
               begin
                  if Decl.Kind = Semantic_IR.Types.Kind_Function_Decl then
                     declare
                        Func_Out : STUNIR.Emitters.CodeGen.IR_Code_Buffer;
                        Func_Ok  : Boolean;
                        F : constant Semantic_IR.Declarations.Function_Declaration := Decl.Func;
                     begin
                        Emit_Function (Self, F, Nodes, Func_Out, Func_Ok);
                        if Func_Ok then
                           STUNIR.Emitters.CodeGen.Append_Raw (Gen, STUNIR.Emitters.CodeGen.Code_Buffers.To_String (Func_Out), Ok);
                           STUNIR.Emitters.CodeGen.Append_Line (Gen, "", Ok);
                        end if;
                     end;
                  end if;
               end;
            end if;
         end;
      end loop;

      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   procedure Emit_Type
     (Self   : in out Python_Emitter;
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
     (Self   : in out Python_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      Gen : STUNIR.Emitters.CodeGen.Code_Generator;
      Ok  : Boolean;
      Name_Str : constant String := Semantic_IR.Types.Name_Strings.To_String (Func.Base.Decl_Name);
   begin
      STUNIR.Emitters.CodeGen.Initialize (Gen, 4);
      STUNIR.Emitters.CodeGen.Append_Line (Gen, "def " & Name_Str & "():", Ok);
      if not Ok then
         Output := STUNIR.Emitters.CodeGen.Code_Buffers.Null_Bounded_String;
         Success := False;
         return;
      end if;
      STUNIR.Emitters.CodeGen.Increase_Indent (Gen);

      if Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Body_ID) then
         declare
            Stmt_Out : STUNIR.Emitters.CodeGen.IR_Code_Buffer;
            Stmt_Ok  : Boolean;
         begin
            STUNIR.Emitters.AST_Render.Render_Statement (Nodes, Func.Body_ID, Stmt_Out, Stmt_Ok);
            if Stmt_Ok then
               STUNIR.Emitters.CodeGen.Append_Line (Gen, STUNIR.Emitters.CodeGen.Code_Buffers.To_String (Stmt_Out), Ok);
            else
               STUNIR.Emitters.CodeGen.Append_Line (Gen, "pass", Ok);
            end if;
         end;
      else
         STUNIR.Emitters.CodeGen.Append_Line (Gen, "pass", Ok);
      end if;

      STUNIR.Emitters.CodeGen.Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Python;
