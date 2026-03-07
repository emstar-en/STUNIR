-------------------------------------------------------------------------------
--  STUNIR Semantic IR Types Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines the core types used throughout the STUNIR Semantic IR
--  system. It extends the flat IR types with semantic annotations, safety
--  levels, and target categories for multi-target code generation.
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
package Semantic_IR.Types with
   Preelaborate,
   SPARK_Mode => On
is
   --  =========================================================================
   --  Bounded String Types (re-exported from IR.Types for consistency)
   --  =========================================================================

   --  Maximum name length for identifiers
   Max_Name_Length : constant := 128;

   --  Maximum hash length: "sha256:" prefix (7) + 64 hex characters
   Max_Hash_Length : constant := 71;

   --  Maximum filesystem path length
   Max_Path_Length : constant := 256;

   --  Bounded string packages for type-safe string handling
   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Name_Length);
   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Hash_Length);
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Path_Length);

   --  Subtypes for Semantic IR string types
   subtype IR_Name is Name_Strings.Bounded_String;
   subtype IR_Hash is Hash_Strings.Bounded_String;
   subtype IR_Path is Path_Strings.Bounded_String;
   subtype Node_ID is Name_Strings.Bounded_String;

   --  =========================================================================
   --  Primitive Types (matching flat IR)
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
   --  Semantic IR Node Kinds
   --  =========================================================================

   --  Discriminator for Semantic IR AST node types
   --  Extends flat IR node kinds with semantic annotations
   type Semantic_Node_Kind is (
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

   --  =========================================================================
   --  Operators
   --  =========================================================================

   --  Binary operators supported in Semantic IR
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

   --  Unary operators
   type Unary_Operator is (
      Op_Neg, Op_Not, Op_Bit_Not,
      Op_Pre_Inc, Op_Pre_Dec, Op_Post_Inc, Op_Post_Dec,
      Op_Deref, Op_Addr_Of
   );

   --  =========================================================================
   --  Storage and Visibility
   --  =========================================================================

   --  Storage class
   type Storage_Class is (
      Storage_Auto, Storage_Static, Storage_Extern,
      Storage_Register, Storage_Stack, Storage_Heap, Storage_Global
   );

   --  Visibility
   type Visibility_Kind is (
      Vis_Public, Vis_Private, Vis_Protected, Vis_Internal
   );

   --  Mutability
   type Mutability_Kind is (
      Mut_Mutable, Mut_Immutable, Mut_Const
   );

   --  Inline hint for optimization
   type Inline_Hint is (
      Inline_Always, Inline_Never, Inline_Hint, Inline_None
   );

   --  =========================================================================
   --  Semantic Annotations (Semantic IR specific)
   --  =========================================================================

   --  Target categories for multi-target code generation
   type Target_Category is (
      Target_Embedded,           --  Embedded systems (bare metal)
      Target_Realtime,           --  Real-time systems (RTOS)
      Target_Safety_Critical,    --  Safety-critical (DO-178C, ISO 26262)
      Target_Gpu,                --  GPU compute targets
      Target_Wasm,               --  WebAssembly targets
      Target_Native              --  Native CPU targets
   );

   --  Safety levels for DO-178C compliance
   type Safety_Level is (
      Level_None,                --  No safety certification required
      Level_DO178C_D,            --  DO-178C Level D (minimal)
      Level_DO178C_C,            --  DO-178C Level C
      Level_DO178C_B,            --  DO-178C Level B
      Level_DO178C_A             --  DO-178C Level A (highest)
   );

   --  Control flow edge kind for CFG representation
   type CFG_Edge_Kind is (
      Edge_Sequential,           --  Normal sequential flow
      Edge_Branch_True,          --  True branch of conditional
      Edge_Branch_False,         --  False branch of conditional
      Edge_Loop_Back,            --  Loop back edge
      Edge_Exception,            --  Exception/error path
      Edge_Return                --  Return edge
   );

   --  =========================================================================
   --  Type System Extensions
   --  =========================================================================

   --  Kind of type reference for the semantic type system
   type Type_Kind is (
      TK_Primitive,             --  Built-in primitive type
      TK_Array,                 --  Array type
      TK_Pointer,               --  Pointer/reference type
      TK_Struct,                --  Struct/class type
      TK_Function,              --  Function type
      TK_Ref                    --  Named type reference
   );

   --  =========================================================================
   --  Validation Functions
   --  =========================================================================

   --  Check if a node ID is valid (non-empty and reasonable length)
   function Is_Valid_Node_ID (ID : Node_ID) return Boolean
      with Post => (if Is_Valid_Node_ID'Result then
                    Name_Strings.Length (ID) > 2);

   --  Check if a hash is valid (correct length for sha256: prefix + hex)
   function Is_Valid_Hash (H : IR_Hash) return Boolean
      with Post => (if Is_Valid_Hash'Result then
                    Hash_Strings.Length (H) = 71); -- "sha256:" + 64 hex

   --  Check if a node kind represents a literal value
   function Is_Literal_Kind (Kind : Semantic_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal | Kind_Float_Literal |
               Kind_String_Literal | Kind_Bool_Literal);

   --  Check if a node kind represents a declaration
   function Is_Declaration_Kind (Kind : Semantic_Node_Kind) return Boolean is
      (Kind in Kind_Function_Decl | Kind_Type_Decl |
               Kind_Const_Decl | Kind_Var_Decl);

   --  Check if a node kind represents a statement
   function Is_Statement_Kind (Kind : Semantic_Node_Kind) return Boolean is
      (Kind in Kind_Block_Stmt | Kind_Expr_Stmt | Kind_If_Stmt |
               Kind_While_Stmt | Kind_For_Stmt | Kind_Return_Stmt |
               Kind_Break_Stmt | Kind_Continue_Stmt |
               Kind_Var_Decl_Stmt | Kind_Assign_Stmt);

   --  Check if a node kind represents an expression
   function Is_Expression_Kind (Kind : Semantic_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal | Kind_Float_Literal |
               Kind_String_Literal | Kind_Bool_Literal |
               Kind_Var_Ref | Kind_Binary_Expr | Kind_Unary_Expr |
               Kind_Function_Call | Kind_Member_Expr |
               Kind_Array_Access | Kind_Cast_Expr |
               Kind_Ternary_Expr | Kind_Array_Init | Kind_Struct_Init);

end Semantic_IR.Types;