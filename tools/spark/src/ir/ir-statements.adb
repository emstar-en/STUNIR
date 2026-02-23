-- STUNIR IR Statements Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body IR.Statements is

   function Is_Valid_Statement (Stmt : Statement_Node) return Boolean is
   begin
      return Is_Valid_Node_ID (Stmt.Base.ID) and then
             Is_Statement_Kind (Stmt.Kind) and then
             Is_Valid_Hash (Stmt.Base.Hash);
   end Is_Valid_Statement;
   
end IR.Statements;
