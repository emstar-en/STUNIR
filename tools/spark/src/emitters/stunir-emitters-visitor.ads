-- STUNIR Visitor Pattern for IR Traversal
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;

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
      Module  : in     IR_Module) is null
   with
     Pre'Class => Is_Valid_Module (Module);

   procedure On_Module_End
     (Context : in out Visitor_Context;
      Module  : in     IR_Module) is null
   with
     Pre'Class => Is_Valid_Module (Module);

   procedure On_Type_Start
     (Context : in out Visitor_Context;
      T       : in     IR_Type_Def) is null
   with
     Pre'Class => T.Field_Cnt >= 0;

   procedure On_Type_End
     (Context : in out Visitor_Context;
      T       : in     IR_Type_Def) is null
   with
     Pre'Class => T.Field_Cnt >= 0;

   procedure On_Function_Start
     (Context : in out Visitor_Context;
      Func    : in     IR_Function) is null
   with
     Pre'Class => Func.Arg_Cnt >= 0;

   procedure On_Function_End
     (Context : in out Visitor_Context;
      Func    : in     IR_Function) is null
   with
     Pre'Class => Func.Arg_Cnt >= 0;

   -- Main traversal procedure
   procedure Traverse_Module
     (Module  : in     IR_Module;
      Context : in out Visitor_Context'Class;
      Result  :    out Visit_Result)
   with
     Pre  => Is_Valid_Module (Module),
     Post => Result in Visit_Result;

end STUNIR.Emitters.Visitor;
