-- STUNIR Visitor Pattern for IR Traversal
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with IR.Modules;
with IR.Declarations;
with IR.Nodes;
with IR.Statements;
with IR.Expressions;
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
      Module  : in     IR.Modules.IR_Module) is null
   with
     Pre'Class => IR.Modules.Is_Valid_Module (Module);

   procedure On_Module_End
     (Context : in out Visitor_Context;
      Module  : in     IR.Modules.IR_Module) is null
   with
     Pre'Class => IR.Modules.Is_Valid_Module (Module);

   procedure On_Type_Start
     (Context : in out Visitor_Context;
      T       : in     IR.Declarations.Type_Declaration) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (T.Base.Node_ID);

   procedure On_Type_End
     (Context : in out Visitor_Context;
      T       : in     IR.Declarations.Type_Declaration) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (T.Base.Node_ID);

   procedure On_Function_Start
     (Context : in out Visitor_Context;
      Func    : in     IR.Declarations.Function_Declaration) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (Func.Base.Node_ID);

   procedure On_Function_End
     (Context : in out Visitor_Context;
      Func    : in     IR.Declarations.Function_Declaration) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (Func.Base.Node_ID);

   procedure On_Statement
     (Context : in out Visitor_Context;
      Stmt    : in     IR.Statements.Statement_Node) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (Stmt.Node_ID);

   procedure On_Expression
     (Context : in out Visitor_Context;
      Expr    : in     IR.Expressions.Expression_Node) is null
   with
     Pre'Class => IR.Nodes.Is_Valid_Node_ID (Expr.Node_ID);

   -- Main traversal procedure
   procedure Traverse_Module
     (Module  : in     IR.Modules.IR_Module;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Context : in out Visitor_Context'Class;
      Result  :    out Visit_Result)
   with
     Pre  => IR.Modules.Is_Valid_Module (Module),
     Post => Result in Visit_Result;

end STUNIR.Emitters.Visitor;
