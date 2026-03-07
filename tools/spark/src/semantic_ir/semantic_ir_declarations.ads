-------------------------------------------------------------------------------
--  STUNIR Semantic IR Declarations Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines declaration nodes for the STUNIR Semantic IR.
--  Declarations include functions, types, constants, and variables.
--
--  Key features:
--  - Explicit type bindings (not just type names)
--  - Visibility and mutability annotations
--  - Safety level annotations for DO-178C compliance
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Declarations with
   SPARK_Mode => On
is
   --  =========================================================================
   --  Function Declaration
   --  =========================================================================

   --  Maximum parameters per function (reduced for stack usage)
   Max_Params : constant := 8;
   Max_Local_Vars : constant := 16;

   --  Parameter structure
   type Param_Record is record
      Name       : IR_Name;        --  Parameter name
      Param_Type : Type_Reference; --  Parameter type (with binding)
      Is_Mutable : Boolean := False;  --  Is this a var/out parameter?
   end record;

   type Param_List is array (1 .. Max_Params) of Param_Record;

   --  Function declaration node
   type Function_Decl is record
      --  Base node information
      Base            : Semantic_Node (Kind_Function_Decl);

      --  Function identification
      Func_Name       : IR_Name;
      Func_Binding    : Node_ID;  --  Self-reference for binding

      --  Type information
      Return_Type     : Type_Reference;

      --  Parameters
      Param_Count     : Natural range 0 .. Max_Params := 0;
      Params          : Param_List;

      --  Body reference (node ID of function body block)
      Body_Node       : Node_ID;

      --  Visibility and linkage
      Visibility      : Visibility_Kind := Vis_Public;
      Storage         : Storage_Class := Storage_Auto;
      Inline_Hint     : Inline_Hint := Inline_None;

      --  Semantic annotations
      Is_Entry_Point  : Boolean := False;
      Safety_Annot    : Safety_Level := Level_None;
   end record;

   --  =========================================================================
   --  Type Declaration
   --  =========================================================================

   --  Maximum fields per struct (reduced for stack usage)
   Max_Fields : constant := 16;

   --  Field record for struct types
   type Field_Record is record
      Name       : IR_Name;
      Field_Type : Type_Reference;
      Offset     : Natural := 0;  --  Byte offset in struct
      Is_Mutable : Boolean := True;
   end record;

   type Field_List is array (1 .. Max_Fields) of Field_Record;

   --  Type declaration node
   type Type_Decl is record
      --  Base node information
      Base            : Semantic_Node (Kind_Type_Decl);

      --  Type identification
      Type_Name       : IR_Name;
      Type_Binding    : Node_ID;  --  Self-reference for binding

      --  Type kind (struct, enum, alias, etc.)
      Is_Struct       : Boolean := False;
      Is_Enum         : Boolean := False;
      Is_Alias        : Boolean := False;

      --  Struct fields (if Is_Struct)
      Field_Count     : Natural range 0 .. Max_Fields := 0;
      Fields          : Field_List;

      --  Underlying type (for aliases)
      Underlying_Type : Type_Reference;

      --  Size information
      Size_In_Bits    : Natural := 0;
      Alignment       : Natural := 0;

      --  Visibility
      Visibility      : Visibility_Kind := Vis_Public;
   end record;

   --  =========================================================================
   --  Constant Declaration
   --  =========================================================================

   --  Constant declaration node
   type Const_Decl is record
      --  Base node information
      Base            : Semantic_Node (Kind_Const_Decl);

      --  Constant identification
      Const_Name      : IR_Name;
      Const_Binding   : Node_ID;  --  Self-reference for binding

      --  Type information
      Const_Type      : Type_Reference;

      --  Initializer (node ID of initializer expression)
      Initializer     : Node_ID;

      --  Visibility
      Visibility      : Visibility_Kind := Vis_Public;
   end record;

   --  =========================================================================
   --  Variable Declaration
   --  =========================================================================

   --  Variable declaration node
   type Var_Decl is record
      --  Base node information
      Base            : Semantic_Node (Kind_Var_Decl);

      --  Variable identification
      Var_Name        : IR_Name;
      Var_Binding     : Node_ID;  --  Self-reference for binding

      --  Type information
      Var_Type        : Type_Reference;

      --  Initializer (node ID of initializer expression, empty if none)
      Initializer     : Node_ID;

      --  Storage and visibility
      Storage         : Storage_Class := Storage_Auto;
      Visibility      : Visibility_Kind := Vis_Private;
      Mutability      : Mutability_Kind := Mut_Mutable;

      --  Semantic annotations
      Is_Volatile     : Boolean := False;
      Is_Atomic       : Boolean := False;
   end record;

   --  =========================================================================
   --  Validation Functions
   --  =========================================================================

   --  Check if a function declaration is valid
   function Is_Valid_Function_Decl (F : Function_Decl) return Boolean
      with Post => (if Is_Valid_Function_Decl'Result then
                    Is_Valid_Semantic_Node (F.Base) and then
                    Name_Strings.Length (F.Func_Name) > 0);

   --  Check if a type declaration is valid
   function Is_Valid_Type_Decl (T : Type_Decl) return Boolean
      with Post => (if Is_Valid_Type_Decl'Result then
                    Is_Valid_Semantic_Node (T.Base) and then
                    Name_Strings.Length (T.Type_Name) > 0);

   --  Check if a constant declaration is valid
   function Is_Valid_Const_Decl (C : Const_Decl) return Boolean
      with Post => (if Is_Valid_Const_Decl'Result then
                    Is_Valid_Semantic_Node (C.Base) and then
                    Name_Strings.Length (C.Const_Name) > 0);

   --  Check if a variable declaration is valid
   function Is_Valid_Var_Decl (V : Var_Decl) return Boolean
      with Post => (if Is_Valid_Var_Decl'Result then
                    Is_Valid_Semantic_Node (V.Base) and then
                    Name_Strings.Length (V.Var_Name) > 0);

end Semantic_IR.Declarations;