-- STUNIR IR Expressions Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with IR.Types; use IR.Types;
with IR.Nodes; use IR.Nodes;

package IR.Expressions with
   SPARK_Mode => On
is
   --  Expression node: composition (IR_Node is not tagged, cannot extend with record)
   type Expression_Node (Kind : IR_Node_Kind) is record
      Base      : IR_Node (Kind);
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
   Max_Arguments : constant := 8;
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

   -- Cast expression
   type Cast_Expression is record
      Base        : Expression_Node (Kind_Cast_Expr);
      Operand_ID  : Node_ID;
      Target_Type : Type_Reference;
   end record;

   -- Array initialization expression
   Max_Array_Elements : constant := 8;
   type Element_ID_List is array (1 .. Max_Array_Elements) of Node_ID;

   type Array_Init_Expression is record
      Base        : Expression_Node (Kind_Array_Init);
      Elem_Count  : Natural range 0 .. Max_Array_Elements := 0;
      Elements    : Element_ID_List;
   end record;

   -- Struct initialization expression
   Max_Struct_Fields : constant := 8;
   type Struct_Field is record
      Field_Name : IR_Name;
      Value_ID   : Node_ID;
   end record;
   type Struct_Field_List is array (1 .. Max_Struct_Fields) of Struct_Field;

   type Struct_Init_Expression is record
      Base        : Expression_Node (Kind_Struct_Init);
      Struct_Name : IR_Name;
      Field_Count : Natural range 0 .. Max_Struct_Fields := 0;
      Fields      : Struct_Field_List;
   end record;

   -- Ternary expression (condition ? then_expr : else_expr)
   type Ternary_Expression is record
      Base        : Expression_Node (Kind_Ternary_Expr);
      Condition_ID: Node_ID;
      Then_ID     : Node_ID;
      Else_ID     : Node_ID;
   end record;
   
   -- Expression validation
   function Is_Expression_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal .. Kind_Struct_Init);
   
   function Is_Valid_Expression (Expr : Expression_Node) return Boolean
      with Post => (if Is_Valid_Expression'Result then
                       Is_Valid_Node_ID (Expr.Base.ID));  --  .Base is IR_Node
   
end IR.Expressions;
