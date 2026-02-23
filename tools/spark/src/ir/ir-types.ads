-------------------------------------------------------------------------------
--  STUNIR IR Types Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines the core types used throughout the STUNIR IR
--  system. It provides bounded string types for memory safety, primitive type
--  enumerations, and node kind discriminators for the IR AST.
--
--  All types are designed for use in SPARK_Mode with formal verification.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

--  NOTE: pragma Pure removed — Ada.Strings.Bounded is not a pure unit.
--  Preelaborate is used instead (allows bounded string instantiations).
package IR.Types with
   Preelaborate,
   SPARK_Mode => On
is
   --  =========================================================================
   --  Bounded String Types
   --  =========================================================================

   --  Maximum name length for identifiers (reduced from 256 to 128 in v0.8.6
   --  to lower stack usage while maintaining usability)
   Max_Name_Length : constant := 128;

   --  Maximum hash length: "sha256:" prefix (7) + 64 hex characters
   Max_Hash_Length : constant := 71;

   --  Maximum filesystem path length
   Max_Path_Length : constant := 256;

   --  Bounded string packages for type-safe string handling
   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Name_Length);
   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Hash_Length);
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Path_Length);

   --  Subtypes for IR string types
   subtype IR_Name is Name_Strings.Bounded_String;
   subtype IR_Hash is Hash_Strings.Bounded_String;
   subtype IR_Path is Path_Strings.Bounded_String;
   subtype Node_ID is Name_Strings.Bounded_String;

   --  =========================================================================
   --  Primitive Types
   --  =========================================================================

   --  Enumeration of all primitive types supported by STUNIR
   type IR_Primitive_Type is (
      Type_Void,                    --  No return value
      Type_Bool,                    --  Boolean (true/false)
      Type_I8, Type_I16, Type_I32, Type_I64,   --  Signed integers
      Type_U8, Type_U16, Type_U32, Type_U64,   --  Unsigned integers
      Type_F32, Type_F64,           --  Floating point (32/64 bit)
      Type_String,                  --  String type
      Type_Char                     --  Character type
   );

   --  =========================================================================
   --  AST Node Kinds
   --  =========================================================================

   --  Discriminator for IR AST node types
   type IR_Node_Kind is (
      --  Module container
      Kind_Module,

      --  Declarations
      Kind_Function_Decl,           --  Function declaration
      Kind_Type_Decl,               --  Type definition
      Kind_Const_Decl,              --  Constant declaration
      Kind_Var_Decl,                --  Variable declaration

      --  Statements
      Kind_Block_Stmt,              --  Block statement { ... }
      Kind_Expr_Stmt,               --  Expression statement
      Kind_If_Stmt,                 --  If statement
      Kind_While_Stmt,              --  While loop
      Kind_For_Stmt,                --  For loop
      Kind_Return_Stmt,             --  Return statement
      Kind_Break_Stmt,              --  Break statement
      Kind_Continue_Stmt,           --  Continue statement
      Kind_Var_Decl_Stmt,           --  Variable declaration statement
      Kind_Assign_Stmt,             --  Assignment statement

      --  Expressions
      Kind_Integer_Literal,         --  Integer literal
      Kind_Float_Literal,           --  Float literal
      Kind_String_Literal,          --  String literal
      Kind_Bool_Literal,            --  Boolean literal
      Kind_Var_Ref,                 --  Variable reference
      Kind_Binary_Expr,             --  Binary expression (a + b)
      Kind_Unary_Expr,              --  Unary expression (-a, !b)
      Kind_Function_Call,           --  Function call
      Kind_Member_Expr,             --  Member access (obj.field)
      Kind_Array_Access,            --  Array indexing (arr[i])
      Kind_Cast_Expr,               --  Type cast
      Kind_Ternary_Expr,            --  Ternary conditional (a ? b : c)
      Kind_Array_Init,              --  Array initialization
      Kind_Struct_Init              --  Struct initialization
   );

   --  NOTE: Parse_* functions moved to end of package (after all type defs)
   --  to avoid forward-reference errors.

   --  =========================================================================
   --  Operators
   --  =========================================================================

   --  Binary operators supported in IR
   type Binary_Operator is (
      --  Arithmetic
      Op_Add, Op_Sub, Op_Mul, Op_Div, Op_Mod,

      --  Comparison
      Op_Eq, Op_Neq, Op_Lt, Op_Leq, Op_Gt, Op_Geq,

      --  Logical
      Op_And, Op_Or,

      --  Bitwise
      Op_Bit_And, Op_Bit_Or, Op_Bit_Xor, Op_Shl, Op_Shr,

      --  Assignment
      Op_Assign
   );
   
   -- Unary operators
   type Unary_Operator is (
      Op_Neg, Op_Not, Op_Bit_Not,
      Op_Pre_Inc, Op_Pre_Dec, Op_Post_Inc, Op_Post_Dec,
      Op_Deref, Op_Addr_Of
   );
   
   -- Storage class
   type Storage_Class is (
      Storage_Auto, Storage_Static, Storage_Extern,
      Storage_Register, Storage_Stack, Storage_Heap, Storage_Global
   );
   
   -- Visibility
   type Visibility_Kind is (
      Vis_Public, Vis_Private, Vis_Protected, Vis_Internal
   );
   
   -- Mutability
   type Mutability_Kind is (
      Mut_Mutable, Mut_Immutable, Mut_Const
   );
   
   -- Inline hint (Inline_Hint_Suggest renamed to avoid conflict with type name)
   type Inline_Hint is (
      Inline_Always, Inline_Never, Inline_Hint_Suggest, Inline_None
   );
   
   -- Source location
   type Source_Location is record
      File   : IR_Path;
      Line   : Positive := 1;
      Column : Positive := 1;
      Length : Natural := 0;
   end record;
   
   -- Target categories (subset for safety)
   type Target_Category is (
      Target_Embedded,
      Target_Realtime,
      Target_Safety_Critical,
      Target_GPU,
      Target_WASM,
      Target_Native,
      Target_JIT,
      Target_Interpreter,
      Target_Functional,
      Target_Logic,
      Target_Constraint,
      Target_Dataflow,
      Target_Reactive,
      Target_Quantum,
      Target_Neuromorphic,
      Target_Biocomputing,
      Target_Molecular,
      Target_Optical,
      Target_Reversible,
      Target_Analog,
      Target_Stochastic,
      Target_Fuzzy,
      Target_Approximate,
      Target_Probabilistic
   );
   
   -- Safety level
   type Safety_Level is (
      Level_None,
      Level_DO178C_D,
      Level_DO178C_C,
      Level_DO178C_B,
      Level_DO178C_A
   );
   
   --  Helper functions
   function Operator_Symbol (Op : Binary_Operator) return String
      with Post => Operator_Symbol'Result'Length > 0;

   function Operator_Symbol (Op : Unary_Operator) return String
      with Post => Operator_Symbol'Result'Length > 0;

   function Primitive_Type_Name (T : IR_Primitive_Type) return String
      with Post => Primitive_Type_Name'Result'Length > 0;

   --  Map JSON schema kind strings to internal enum values
   --  (declared here, after all types, to avoid forward-reference errors)
   function Parse_Node_Kind       (Kind_Str    : String) return IR_Node_Kind    with Global => null;
   function Parse_Binary_Operator (Op_Str      : String) return Binary_Operator with Global => null;
   function Parse_Unary_Operator  (Op_Str      : String) return Unary_Operator  with Global => null;
   function Parse_Visibility      (Vis_Str     : String) return Visibility_Kind  with Global => null;
   function Parse_Mutability      (Mut_Str     : String) return Mutability_Kind  with Global => null;
   function Parse_Storage_Class   (Storage_Str : String) return Storage_Class   with Global => null;
   function Parse_Inline_Hint     (Inline_Str  : String) return Inline_Hint     with Global => null;
   function Parse_Primitive_Type  (Prim_Str    : String) return IR_Primitive_Type with Global => null;
   function Parse_Target_Category (Cat_Str     : String) return Target_Category  with Global => null;
   function Parse_Safety_Level    (Level_Str   : String) return Safety_Level     with Global => null;

end IR.Types;
