-- STUNIR Semantic IR Types Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package Semantic_IR.Types with
   Pure,
   SPARK_Mode => On
is
   -- Bounded string types for safety
   --  v0.8.6: Reduced from 256 to 128 to lower stack usage
   Max_Name_Length : constant := 128;
   Max_Hash_Length : constant := 71; -- "sha256:" + 64 hex chars
   Max_Path_Length : constant := 256;
   
   package Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Name_Length);
   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Hash_Length);
   package Path_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Path_Length);
   
   subtype IR_Name is Name_Strings.Bounded_String;
   subtype IR_Hash is Hash_Strings.Bounded_String;
   subtype IR_Path is Path_Strings.Bounded_String;
   subtype Node_ID is Name_Strings.Bounded_String;
   
   -- Primitive types enumeration
   type IR_Primitive_Type is (
      Type_Void,
      Type_Bool,
      Type_I8, Type_I16, Type_I32, Type_I64,
      Type_U8, Type_U16, Type_U32, Type_U64,
      Type_F32, Type_F64,
      Type_String,
      Type_Char
   );
   
   -- Node kind discriminator
   type IR_Node_Kind is (
      -- Module
      Kind_Module,
      -- Declarations
      Kind_Function_Decl,
      Kind_Type_Decl,
      Kind_Const_Decl,
      Kind_Var_Decl,
      -- Statements
      Kind_Block_Stmt,
      Kind_Expr_Stmt,
      Kind_If_Stmt,
      Kind_While_Stmt,
      Kind_For_Stmt,
      Kind_Return_Stmt,
      Kind_Break_Stmt,
      Kind_Continue_Stmt,
      Kind_Var_Decl_Stmt,
      Kind_Assign_Stmt,
      -- Expressions
      Kind_Integer_Literal,
      Kind_Float_Literal,
      Kind_String_Literal,
      Kind_Bool_Literal,
      Kind_Var_Ref,
      Kind_Binary_Expr,
      Kind_Unary_Expr,
      Kind_Function_Call,
      Kind_Member_Expr,
      Kind_Array_Access,
      Kind_Cast_Expr,
      Kind_Ternary_Expr,
      Kind_Array_Init,
      Kind_Struct_Init
   );
   
   -- Binary operators
   type Binary_Operator is (
      Op_Add, Op_Sub, Op_Mul, Op_Div, Op_Mod,
      Op_Eq, Op_Neq, Op_Lt, Op_Leq, Op_Gt, Op_Geq,
      Op_And, Op_Or,
      Op_Bit_And, Op_Bit_Or, Op_Bit_Xor, Op_Shl, Op_Shr,
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
   
   -- Inline hint
   type Inline_Hint is (
      Inline_Always, Inline_Never, Inline_Hint, Inline_None
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
      Target_Functional
   );
   
   -- Safety level
   type Safety_Level is (
      Level_None,
      Level_DO178C_D,
      Level_DO178C_C,
      Level_DO178C_B,
      Level_DO178C_A
   );
   
   -- Helper functions
   function Operator_Symbol (Op : Binary_Operator) return String
      with Post => Operator_Symbol'Result'Length > 0;
   
   function Operator_Symbol (Op : Unary_Operator) return String
      with Post => Operator_Symbol'Result'Length > 0;
   
   function Primitive_Type_Name (T : IR_Primitive_Type) return String
      with Post => Primitive_Type_Name'Result'Length > 0;
      
end Semantic_IR.Types;
