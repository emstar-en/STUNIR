-- STUNIR Base Emitter Interface
-- DO-178C Level A
-- Phase 3a: Core Category Emitters
--
-- NOTE: STUNIR.Emitters.CodeGen and STUNIR.Emitters.Node_Table are child
-- packages of this package. In Ada, a parent package spec CANNOT with its
-- own children (circular dependency). Those withs are in the body only.
-- The abstract interface uses Ada.Strings.Bounded directly for the output
-- buffer type, and the IR types for module/declaration parameters.

with Ada.Strings.Bounded;
with IR.Modules;
with IR.Declarations;
with IR.Nodes;
with IR.Types;

package STUNIR.Emitters is
   pragma SPARK_Mode (On);

   subtype Target_Category is IR.Types.Target_Category;

   type Emitter_Status is
     (Status_Success, Status_Error_Parse, Status_Error_Generate, Status_Error_IO);

   --  NOTE: Code_Buffers and IR_Code_Buffer are defined in STUNIR.Emitters.CodeGen
   --  (the child package). They are NOT defined here to avoid duplicate instances.
   --  NOTE: Max_Nodes and Node_Index are defined in STUNIR.Emitters.Node_Table
   --  (the child package). They are NOT defined here to avoid duplicate types.
   --  The Ada.Strings.Bounded with clause is kept for potential future use.

   --  Abstract emitter base type
   type Base_Emitter is abstract tagged record
      Category : Target_Category := IR.Types.Target_Embedded;
      Status   : Emitter_Status  := Status_Success;
   end record;

   --  NOTE: Abstract Emit_Module/Emit_Type/Emit_Function are NOT declared here.
   --  Reason: those procedures take STUNIR.Emitters.Node_Table.Node_Table and
   --  STUNIR.Emitters.CodeGen.IR_Code_Buffer as parameters. A parent package
   --  spec cannot with its own children (Ada circular dependency rule).
   --
   --  Instead, each child emitter package (CFamily, Python, Lisp, etc.) defines
   --  its own concrete Emit_Module procedure with the full Node_Table parameter.
   --  code_emitter.adb dispatches statically by target name, not via class-wide
   --  dispatch through this base type. The base type exists only to carry the
   --  Category and Status fields shared by all emitters.
   --
   --  If class-wide dispatch is needed in the future, move the abstract interface
   --  to STUNIR.Emitters.CodeGen (which already withs Node_Table) or introduce
   --  a separate STUNIR.Emitters.Interface package that withs both children.

   --  Common utility functions
   function Get_Category_Name (Cat : Target_Category) return String
   with
     Global => null,
     Post   => Get_Category_Name'Result'Length > 0;

   function Get_Status_Name (Status : Emitter_Status) return String
   with
     Global => null,
     Post   => Get_Status_Name'Result'Length > 0;

end STUNIR.Emitters;
