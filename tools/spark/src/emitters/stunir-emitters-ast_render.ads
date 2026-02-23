-------------------------------------------------------------------------------
--  STUNIR AST Render Helpers
--  DO-178C Level A
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR.Emitters.CodeGen;
with STUNIR.Emitters.Node_Table;
with IR.Types;
with IR.Nodes;
with IR.Statements;
with IR.Expressions;

package STUNIR.Emitters.AST_Render is
   pragma SPARK_Mode (On);

   procedure Render_Expression
     (Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Expr_ID: in     IR.Types.Node_ID;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

   procedure Render_Statement
     (Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Stmt_ID: in     IR.Types.Node_ID;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean);

end STUNIR.Emitters.AST_Render;
