-- STUNIR Semantic IR Expressions Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Expressions is

   function Is_Valid_Expression (Expr : Expression_Node) return Boolean is
   begin
      return Is_Valid_Node_ID (Expr.Base.ID) and then
             Is_Expression_Kind (Expr.Kind) and then
             Is_Valid_Hash (Expr.Base.Hash);
   end Is_Valid_Expression;
   
end Semantic_IR.Expressions;
