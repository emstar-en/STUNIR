-- STUNIR Semantic IR Declarations Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Declarations is

   function Is_Valid_Declaration (Decl : Declaration_Node) return Boolean is
      use Name_Strings;
   begin
      return Is_Valid_Node_ID (Decl.Node_ID) and then
             Is_Declaration_Kind (Decl.Kind) and then
             Is_Valid_Hash (Decl.Hash) and then
             Length (Decl.Decl_Name) > 0;
   end Is_Valid_Declaration;
   
end Semantic_IR.Declarations;
