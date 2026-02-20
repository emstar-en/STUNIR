-- STUNIR Semantic IR Statements Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Statements with
   SPARK_Mode => On
is
   -- Statement types
   type Statement_Node (Kind : IR_Node_Kind) is new IR_Node (Kind) with null record;
   
   -- Block statement
   Max_Statements : constant := 128;
   type Statement_List is array (1 .. Max_Statements) of Node_ID;
   
   type Block_Statement is record
      Base       : Statement_Node (Kind_Block_Stmt);
      Stmt_Count : Natural range 0 .. Max_Statements := 0;
      Statements : Statement_List;
      Scope_ID   : IR_Name;
   end record;

   -- Expression statement
   type Expr_Statement is record
      Base       : Statement_Node (Kind_Expr_Stmt);
      Expr_ID    : Node_ID;
   end record;
   
   -- If statement
   type If_Statement is record
      Base         : Statement_Node (Kind_If_Stmt);
      Condition_ID : Node_ID;
      Then_ID      : Node_ID;
      Else_ID      : Node_ID; -- Can be empty
   end record;
   
   -- While statement
   type While_Statement is record
      Base         : Statement_Node (Kind_While_Stmt);
      Condition_ID : Node_ID;
      Body_ID      : Node_ID;
      Loop_Bound   : Natural := 0; -- 0 means unbounded
      Unroll       : Boolean := False;
   end record;
   
   -- For statement
   type For_Statement is record
      Base         : Statement_Node (Kind_For_Stmt);
      Init_ID      : Node_ID;
      Condition_ID : Node_ID;
      Increment_ID : Node_ID;
      Body_ID      : Node_ID;
      Loop_Bound   : Natural := 0;
      Unroll       : Boolean := False;
      Vectorize    : Boolean := False;
   end record;
   
   -- Return statement
   type Return_Statement is record
      Base     : Statement_Node (Kind_Return_Stmt);
      Value_ID : Node_ID; -- Can be empty for void returns
   end record;

   -- Break statement
   type Break_Statement is record
      Base : Statement_Node (Kind_Break_Stmt);
   end record;

   -- Continue statement
   type Continue_Statement is record
      Base : Statement_Node (Kind_Continue_Stmt);
   end record;
   
   -- Variable declaration statement
   type Var_Decl_Statement is record
      Base         : Statement_Node (Kind_Var_Decl_Stmt);
      Var_Name     : IR_Name;
      Var_Type     : Type_Reference;
      Init_ID      : Node_ID; -- Initializer expression
      Storage      : Storage_Class := Storage_Auto;
      Mutability   : Mutability_Kind := Mut_Mutable;
   end record;
   
   -- Assignment statement
   type Assign_Statement is record
      Base      : Statement_Node (Kind_Assign_Stmt);
      Target_ID : Node_ID; -- LValue expression
      Value_ID  : Node_ID; -- RValue expression
   end record;
   
   -- Statement validation
   function Is_Statement_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Block_Stmt .. Kind_Assign_Stmt);
   
   function Is_Valid_Statement (Stmt : Statement_Node) return Boolean
      with Post => (if Is_Valid_Statement'Result then
                       Is_Valid_Node_ID (Stmt.Node_ID));
   
end Semantic_IR.Statements;
