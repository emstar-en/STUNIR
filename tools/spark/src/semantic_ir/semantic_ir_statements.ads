-------------------------------------------------------------------------------
--  STUNIR Semantic IR Statements Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines statement nodes for the STUNIR Semantic IR.
--  Statements include blocks, conditionals, loops, and control flow.
--
--  Key features:
--  - Control flow graph edges for CFG representation
--  - Safety annotations for DO-178C compliance
--  - Normalized form with explicit control flow
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Statements with
   SPARK_Mode => On
is
   --  =========================================================================
   --  Block Statement
   --  =========================================================================

   --  Maximum statements per block (reduced for stack usage)
   Max_Block_Stmts : constant := 32;

   type Stmt_List is array (1 .. Max_Block_Stmts) of Node_ID;

   --  Block statement node
   type Block_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Block_Stmt);

      --  Statements in the block
      Stmt_Count      : Natural range 0 .. Max_Block_Stmts := 0;
      Statements      : Stmt_List;

      --  Scope information
      Parent_Scope    : Node_ID;  --  Parent block (empty if top-level)
   end record;

   --  =========================================================================
   --  Expression Statement
   --  =========================================================================

   --  Expression statement node
   type Expr_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Expr_Stmt);

      --  Expression
      Expr_Node       : Node_ID;  --  Reference to expression node
   end record;

   --  =========================================================================
   --  If Statement
   --  =========================================================================

   --  If statement node
   type If_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_If_Stmt);

      --  Condition
      Condition       : Node_ID;  --  Reference to condition expression

      --  Branches
      Then_Branch     : Node_ID;  --  Reference to then block
      Else_Branch     : Node_ID;  --  Reference to else block (empty if none)
   end record;

   --  =========================================================================
   --  While Statement
   --  =========================================================================

   --  While statement node
   type While_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_While_Stmt);

      --  Condition
      Condition       : Node_ID;  --  Reference to condition expression

      --  Body
      Body_Node       : Node_ID;  --  Reference to body block

      --  Control flow graph edges
      Loop_Back_Edge  : CFG_Edge;  --  Edge back to loop start
   end record;

   --  =========================================================================
   --  For Statement
   --  =========================================================================

   --  For statement node
   type For_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_For_Stmt);

      --  Loop variable
      Loop_Var        : Node_ID;  --  Reference to variable declaration

      --  Range
      Start_Expr      : Node_ID;  --  Start expression
      End_Expr        : Node_ID;  --  End expression
      Step_Expr       : Node_ID;  --  Step expression (empty if 1)

      --  Body
      Body_Node       : Node_ID;  --  Reference to body block

      --  Control flow graph edges
      Loop_Back_Edge  : CFG_Edge;  --  Edge back to loop start
   end record;

   --  =========================================================================
   --  Return Statement
   --  =========================================================================

   --  Return statement node
   type Return_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Return_Stmt);

      --  Return value (empty if void)
      Return_Value    : Node_ID;  --  Reference to return expression
   end record;

   --  =========================================================================
   --  Break Statement
   --  =========================================================================

   --  Break statement node
   type Break_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Break_Stmt);

      --  Target loop (for nested loops)
      Target_Loop     : Node_ID;  --  Reference to enclosing loop
   end record;

   --  =========================================================================
   --  Continue Statement
   --  =========================================================================

   --  Continue statement node
   type Continue_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Continue_Stmt);

      --  Target loop (for nested loops)
      Target_Loop     : Node_ID;  --  Reference to enclosing loop
   end record;

   --  =========================================================================
   --  Variable Declaration Statement
   --  =========================================================================

   --  Variable declaration statement node
   type Var_Decl_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Var_Decl_Stmt);

      --  Variable declaration
      Var_Decl_Node   : Node_ID;  --  Reference to variable declaration
   end record;

   --  =========================================================================
   --  Assignment Statement
   --  =========================================================================

   --  Assignment statement node
   type Assign_Stmt is record
      --  Base node information
      Base            : Semantic_Node (Kind_Assign_Stmt);

      --  Target and value
      Target          : Node_ID;  --  Reference to target (lvalue)
      Value           : Node_ID;  --  Reference to value expression
   end record;

   --  =========================================================================
   --  Validation Functions
   --  =========================================================================

   --  Check if a block statement is valid
   function Is_Valid_Block_Stmt (B : Block_Stmt) return Boolean
      with Post => (if Is_Valid_Block_Stmt'Result then
                    Is_Valid_Semantic_Node (B.Base));

   --  Check if an if statement is valid
   function Is_Valid_If_Stmt (I : If_Stmt) return Boolean
      with Post => (if Is_Valid_If_Stmt'Result then
                    Is_Valid_Semantic_Node (I.Base) and then
                    Is_Valid_Node_ID (I.Condition) and then
                    Is_Valid_Node_ID (I.Then_Branch));

   --  Check if a while statement is valid
   function Is_Valid_While_Stmt (W : While_Stmt) return Boolean
      with Post => (if Is_Valid_While_Stmt'Result then
                    Is_Valid_Semantic_Node (W.Base) and then
                    Is_Valid_Node_ID (W.Condition) and then
                    Is_Valid_Node_ID (W.Body_Node));

   --  Check if a for statement is valid
   function Is_Valid_For_Stmt (F : For_Stmt) return Boolean
      with Post => (if Is_Valid_For_Stmt'Result then
                    Is_Valid_Semantic_Node (F.Base) and then
                    Is_Valid_Node_ID (F.Loop_Var) and then
                    Is_Valid_Node_ID (F.Start_Expr) and then
                    Is_Valid_Node_ID (F.End_Expr) and then
                    Is_Valid_Node_ID (F.Body_Node));

end Semantic_IR.Statements;