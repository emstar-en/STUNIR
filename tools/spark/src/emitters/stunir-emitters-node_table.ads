-- STUNIR Emitter Node Table (Semantic IR AST storage)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;
with Semantic_IR.Declarations; use Semantic_IR.Declarations;
with Semantic_IR.Statements; use Semantic_IR.Statements;
with Semantic_IR.Expressions; use Semantic_IR.Expressions;

package STUNIR.Emitters.Node_Table is

   -- Maximum nodes stored for emission
   Max_Nodes : constant := 2048;
   type Node_Index is range 0 .. Max_Nodes;

   type Node_Kind_Group is (Group_Node, Group_Declaration, Group_Statement, Group_Expression);

   type Declaration_Record (Kind : IR_Node_Kind := Kind_Function_Decl) is record
      case Kind is
         when Kind_Function_Decl =>
            Func : Function_Declaration;
         when Kind_Type_Decl =>
            Typ  : Type_Declaration;
         when Kind_Const_Decl =>
            Cst  : Const_Declaration;
         when Kind_Var_Decl =>
            Var  : Variable_Declaration;
         when others =>
            Base : Declaration_Node (Kind);
      end case;
   end record;

   type Statement_Record (Kind : IR_Node_Kind := Kind_Block_Stmt) is record
      case Kind is
         when Kind_Block_Stmt =>
            Block : Block_Statement;
         when Kind_Expr_Stmt =>
            Expr  : Expr_Statement;
         when Kind_If_Stmt =>
            If_S  : If_Statement;
         when Kind_While_Stmt =>
            While_S : While_Statement;
         when Kind_For_Stmt =>
            For_S  : For_Statement;
         when Kind_Return_Stmt =>
            Ret    : Return_Statement;
         when Kind_Break_Stmt =>
            Brk    : Break_Statement;
         when Kind_Continue_Stmt =>
            Cont   : Continue_Statement;
         when Kind_Var_Decl_Stmt =>
            Var_D  : Var_Decl_Statement;
         when Kind_Assign_Stmt =>
            Assign : Assign_Statement;
         when others =>
            Base   : Statement_Node (Kind);
      end case;
   end record;

   type Expression_Record (Kind : IR_Node_Kind := Kind_Integer_Literal) is record
      case Kind is
         when Kind_Binary_Expr =>
            Bin : Binary_Expression;
         when Kind_Unary_Expr =>
            Un  : Unary_Expression;
         when Kind_Function_Call =>
            Call : Function_Call;
         when Kind_Member_Expr =>
            Mem  : Member_Expression;
         when Kind_Array_Access =>
            Arr  : Array_Access_Expr;
         when Kind_Cast_Expr =>
            Cast : Cast_Expression;
         when Kind_Ternary_Expr =>
            Ter  : Ternary_Expression;
         when Kind_Array_Init =>
            Arr_Init : Array_Init_Expression;
         when Kind_Struct_Init =>
            Struct_Init : Struct_Init_Expression;
         when others =>
            Base : Expression_Node (Kind);
      end case;
   end record;

   type Node_Record (Group : Node_Kind_Group := Group_Node) is record
      case Group is
         when Group_Node =>
            Node : IR_Node (Kind_Module);
         when Group_Declaration =>
            Decl : Declaration_Record;
         when Group_Statement =>
            Stmt : Statement_Record;
         when Group_Expression =>
            Expr : Expression_Record;
      end case;
   end record;

   type Node_Entry is record
      ID    : Node_ID;
      Kind  : IR_Node_Kind := Kind_Module;
      Group : Node_Kind_Group := Group_Node;
      Data  : Node_Record;
      Used  : Boolean := False;
   end record;

   type Node_Table is record
      Entries : array (1 .. Max_Nodes) of Node_Entry;
      Count   : Node_Index := 0;
   end record;

   procedure Initialize (Table : out Node_Table)
     with Post => Table.Count = 0;

   function Lookup (Table : Node_Table; ID : Node_ID) return Node_Index
     with Post => Lookup'Result in Node_Index;

   function Get_Declaration (Table : Node_Table; Index : Node_Index) return Declaration_Record
     with Pre  => Index > 0 and then Index <= Table.Count and then Table.Entries (Index).Group = Group_Declaration,
          Post => Is_Valid_Node_ID (Get_Declaration'Result.Base.Node_ID);

   function Get_Statement (Table : Node_Table; Index : Node_Index) return Statement_Record
     with Pre  => Index > 0 and then Index <= Table.Count and then Table.Entries (Index).Group = Group_Statement,
          Post => Is_Valid_Node_ID (Get_Statement'Result.Base.Node_ID);

   function Get_Expression (Table : Node_Table; Index : Node_Index) return Expression_Record
     with Pre  => Index > 0 and then Index <= Table.Count and then Table.Entries (Index).Group = Group_Expression,
          Post => Is_Valid_Node_ID (Get_Expression'Result.Base.Node_ID);

   procedure Add_Node
     (Table : in out Node_Table;
      Node  : IR_Node;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Node.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Declaration
     (Table   : in out Node_Table;
      Decl    : Declaration_Record;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Decl.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Function_Declaration
     (Table   : in out Node_Table;
      Decl    : Function_Declaration;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Decl.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Type_Declaration
     (Table   : in out Node_Table;
      Decl    : Type_Declaration;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Decl.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Const_Declaration
     (Table   : in out Node_Table;
      Decl    : Const_Declaration;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Decl.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Variable_Declaration
     (Table   : in out Node_Table;
      Decl    : Variable_Declaration;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Decl.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Statement
     (Table   : in out Node_Table;
      Stmt    : Statement_Record;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Stmt.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Expression
     (Table   : in out Node_Table;
      Expr    : Expression_Record;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Expr.Base.Node_ID),
      Post => Table.Count <= Max_Nodes;

   procedure Add_Expression_Node
     (Table   : in out Node_Table;
      Expr    : Expression_Node;
      Success : out Boolean)
   with
      Pre  => Is_Valid_Node_ID (Expr.Node_ID),
      Post => Table.Count <= Max_Nodes;

end STUNIR.Emitters.Node_Table;
