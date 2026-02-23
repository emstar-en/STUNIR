-------------------------------------------------------------------------------
--  STUNIR AST Render Helpers (Implementation)
--  DO-178C Level A
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR.Emitters.CodeGen;
with STUNIR.Emitters.Node_Table;
with IR.Types;
with IR.Nodes;
with IR.Expressions;
with IR.Statements;

package body STUNIR.Emitters.AST_Render is

   use STUNIR.Emitters.CodeGen;
   use IR.Types;
   use IR.Nodes;
   use IR.Expressions;
   use IR.Statements;

   procedure Append
     (Buffer : in out IR_Code_Buffer;
      Text   : in     String;
      Ok     :    out Boolean)
   is
   begin
      Ok := False;
      if Code_Buffers.Length (Buffer) + Text'Length <= Max_Code_Length then
         Code_Buffers.Append (Buffer, Text);
         Ok := True;
      end if;
   end Append;

   procedure Render_Expression
     (Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Expr_ID: in     Node_ID;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
        STUNIR.Emitters.Node_Table.Lookup (Nodes, Expr_ID);
      Ok : Boolean;
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      if Index = 0 then
         return;
      end if;

      declare
         Expr : constant STUNIR.Emitters.Node_Table.Expression_Record :=
           STUNIR.Emitters.Node_Table.Get_Expression (Nodes, Index);
      begin
         case Expr.Kind is
            when Kind_Integer_Literal =>
               Append (Output, "0", Ok);
            when Kind_Float_Literal =>
               Append (Output, "0.0", Ok);
            when Kind_String_Literal =>
               Append (Output, """""", Ok);
            when Kind_Bool_Literal =>
               Append (Output, "false", Ok);
            when Kind_Var_Ref =>
               Append (Output, "var", Ok);
            when Kind_Binary_Expr =>
               Append (Output, "(", Ok);
               if Ok then
                  Append (Output, "...", Ok);
               end if;
               if Ok then
                  Append (Output, ")", Ok);
               end if;
            when Kind_Function_Call =>
               Append (Output, "call()", Ok);
            when others =>
               Append (Output, "/* expr */", Ok);
         end case;
      end;

      Success := Ok;
   end Render_Expression;

   procedure Render_Statement
     (Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Stmt_ID: in     Node_ID;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
        STUNIR.Emitters.Node_Table.Lookup (Nodes, Stmt_ID);
      Ok : Boolean;
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      if Index = 0 then
         return;
      end if;

      declare
         Stmt : constant STUNIR.Emitters.Node_Table.Statement_Record :=
           STUNIR.Emitters.Node_Table.Get_Statement (Nodes, Index);
      begin
         case Stmt.Kind is
            when Kind_Return_Stmt =>
               Append (Output, "return;", Ok);
            when Kind_Assign_Stmt =>
               Append (Output, "/* assign */", Ok);
            when Kind_Expr_Stmt =>
               Append (Output, "/* expr */", Ok);
            when others =>
               Append (Output, "/* stmt */", Ok);
         end case;
      end;

      Success := Ok;
   end Render_Statement;

end STUNIR.Emitters.AST_Render;
