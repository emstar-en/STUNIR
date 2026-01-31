-- STUNIR Base Emitter Interface
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;

package STUNIR.Emitters is
   pragma SPARK_Mode (On);

   type Target_Category is
     (Category_Embedded, Category_GPU, Category_WASM,
      Category_Assembly, Category_Polyglot);

   type Emitter_Status is
     (Status_Success, Status_Error_Parse, Status_Error_Generate, Status_Error_IO);

   -- Abstract emitter interface
   type Base_Emitter is abstract tagged record
      Category : Target_Category := Category_Embedded;
      Status   : Emitter_Status := Status_Success;
   end record;

   -- Abstract methods (must be overridden by concrete emitters)
   procedure Emit_Module
     (Self   : in out Base_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   procedure Emit_Type
     (Self   : in out Base_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   procedure Emit_Function
     (Self   : in out Base_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is abstract
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

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
