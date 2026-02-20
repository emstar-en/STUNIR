-- STUNIR Visitor Pattern for IR Traversal
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with Semantic_IR.Nodes;
with Semantic_IR.Statements;
with Semantic_IR.Expressions;
with STUNIR.Emitters.Node_Table;

package STUNIR.Emitters.Visitor is
   pragma SPARK_Mode (On);

   type Visit_Result is (Continue, Skip, Abort_Visit);

   -- Visitor callbacks interface
   type Visitor_Context is tagged record
      Result : Visit_Result := Continue;
   end record;

   -- Virtual methods for visitor callbacks
   procedure On_Module_Start
     (Context : in out Visitor_Context;
      Module  : in     Semantic_IR.Modules.IR_Module) is null
   with
     Pre'Class => Semantic_IR.Modules.Is_Valid_Module (Module);

   procedure On_Module_End
     (Context : in out Visitor_Context;
      Module  : in     Semantic_IR.Modules.IR_Module) is null
   with
     Pre'Class => Semantic_IR.Modules.Is_Valid_Module (Module);

   procedure On_Type_Start
     (Context : in out Visitor_Context;
      T       : in     Semantic_IR.Declarations.Type_Declaration) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (T.Base.Node_ID);

   procedure On_Type_End
     (Context : in out Visitor_Context;
      T       : in     Semantic_IR.Declarations.Type_Declaration) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (T.Base.Node_ID);

   procedure On_Function_Start
     (Context : in out Visitor_Context;
      Func    : in     Semantic_IR.Declarations.Function_Declaration) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Base.Node_ID);

   procedure On_Function_End
     (Context : in out Visitor_Context;
      Func    : in     Semantic_IR.Declarations.Function_Declaration) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Base.Node_ID);

   procedure On_Statement
     (Context : in out Visitor_Context;
      Stmt    : in     Semantic_IR.Statements.Statement_Node) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (Stmt.Node_ID);

   procedure On_Expression
     (Context : in out Visitor_Context;
      Expr    : in     Semantic_IR.Expressions.Expression_Node) is null
   with
     Pre'Class => Semantic_IR.Nodes.Is_Valid_Node_ID (Expr.Node_ID);

   -- Main traversal procedure
   procedure Traverse_Module
     (Module  : in     Semantic_IR.Modules.IR_Module;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Context : in out Visitor_Context'Class;
      Result  :    out Visit_Result)
   with
     Pre  => Semantic_IR.Modules.Is_Valid_Module (Module),
     Post => Result in Visit_Result;

end STUNIR.Emitters.Visitor;
