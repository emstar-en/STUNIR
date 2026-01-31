-- STUNIR Semantic IR Expressions Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Expressions with
   SPARK_Mode => On
is
   -- Expression types
   type Expression_Node (Kind : IR_Node_Kind) is new IR_Node (Kind) with record
      Expr_Type : Type_Reference;
   end record;
   
   -- Binary expression record
   type Binary_Expression is record
      Base     : Expression_Node (Kind_Binary_Expr);
      Operator : Binary_Operator;
      -- Left and Right would be references to child nodes (node IDs)
      Left_ID  : Node_ID;
      Right_ID : Node_ID;
   end record;
   
   -- Unary expression record
   type Unary_Expression is record
      Base       : Expression_Node (Kind_Unary_Expr);
      Operator   : Unary_Operator;
      Operand_ID : Node_ID;
   end record;
   
   -- Function call expression
   Max_Arguments : constant := 32;
   type Argument_List is array (1 .. Max_Arguments) of Node_ID;
   
   type Function_Call is record
      Base          : Expression_Node (Kind_Function_Call);
      Function_Name : IR_Name;
      Func_Binding  : Node_ID;
      Arg_Count     : Natural range 0 .. Max_Arguments := 0;
      Arguments     : Argument_List;
   end record;
   
   -- Member access expression
   type Member_Expression is record
      Base        : Expression_Node (Kind_Member_Expr);
      Object_ID   : Node_ID;
      Member_Name : IR_Name;
      Is_Arrow    : Boolean := False;
   end record;
   
   -- Array access expression
   type Array_Access_Expr is record
      Base     : Expression_Node (Kind_Array_Access);
      Array_ID : Node_ID;
      Index_ID : Node_ID;
   end record;
   
   -- Expression validation
   function Is_Expression_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal .. Kind_Struct_Init);
   
   function Is_Valid_Expression (Expr : Expression_Node) return Boolean
      with Post => (if Is_Valid_Expression'Result then
                       Is_Valid_Node_ID (Expr.Node_ID));
   
end Semantic_IR.Expressions;
