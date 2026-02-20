-- STUNIR Base Emitter Interface
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with Semantic_IR.Modules;
with Semantic_IR.Declarations;
with Semantic_IR.Nodes;
with Semantic_IR.Types;
with STUNIR.Emitters.CodeGen;
with STUNIR.Emitters.Node_Table;

package STUNIR.Emitters is
   pragma SPARK_Mode (On);

  subtype Target_Category is Semantic_IR.Types.Target_Category;

   type Emitter_Status is
     (Status_Success, Status_Error_Parse, Status_Error_Generate, Status_Error_IO);

   -- Abstract emitter interface
  type Base_Emitter is abstract tagged record
    Category : Target_Category := Semantic_IR.Types.Target_Embedded;
    Status   : Emitter_Status := Status_Success;
  end record;

   -- Abstract methods (must be overridden by concrete emitters)
   procedure Emit_Module
     (Self   : in out Base_Emitter;
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => Semantic_IR.Modules.Is_Valid_Module (Module),
     Post'Class => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) > 0);

   procedure Emit_Type
     (Self   : in out Base_Emitter;
      T      : in     Semantic_IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => Semantic_IR.Nodes.Is_Valid_Node_ID (T.Base.Node_ID),
     Post'Class => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) >= 0);

   procedure Emit_Function
     (Self   : in out Base_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Base.Node_ID),
     Post'Class => (if Success then STUNIR.Emitters.CodeGen.Code_Buffers.Length (Output) >= 0);

   -- Common utility functions
   function Get_Category_Name (Cat : Target_Category) return String
   with
     Global => null,
     Post => Get_Category_Name'Result'Length > 0;

   function Get_Status_Name (Status : Emitter_Status) return String
   with
     Global => null,
     Post => Get_Status_Name'Result'Length > 0;

end STUNIR.Emitters;
