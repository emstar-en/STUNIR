-------------------------------------------------------------------------------
--  STUNIR Semantic IR Expressions Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines expression nodes for the STUNIR Semantic IR.
--  Expressions include literals, references, operators, and calls.
--
--  Key features:
--  - Explicit type bindings for all expressions
--  - Safety annotations for DO-178C compliance
--  - Normalized form with explicit type conversions
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;

package Semantic_IR.Expressions with
   SPARK_Mode => On
is
   --  =========================================================================
   --  Literal Expressions
   --  =========================================================================

   --  Integer literal expression
   type Integer_Literal_Expr is record
      Base      : Semantic_Node (Kind_Integer_Literal);
      Value     : Long_Long_Integer;
      Radix     : Integer range 2 .. 16 := 10;
      Lit_Type  : Type_Reference;  --  Must be integer type
   end record;

   --  Float literal expression
   type Float_Literal_Expr is record
      Base      : Semantic_Node (Kind_Float_Literal);
      Value     : Long_Float;
      Lit_Type  : Type_Reference;  --  Must be float type
   end record;

   --  String literal expression
   type String_Literal_Expr is record
      Base      : Semantic_Node (Kind_String_Literal);
      Value     : IR_Name;  --  Bounded string
      Lit_Type  : Type_Reference;  --  Must be string type
   end record;

   --  Boolean literal expression
   type Bool_Literal_Expr is record
      Base      : Semantic_Node (Kind_Bool_Literal);
      Value     : Boolean;
      Lit_Type  : Type_Reference;  --  Must be bool type
   end record;

   --  =========================================================================
   --  Reference Expressions
   --  =========================================================================

   --  Variable reference expression
   type Var_Ref_Expr is record
      Base        : Semantic_Node (Kind_Var_Ref);
      Var_Name    : IR_Name;
      Var_Binding : Node_ID;  --  Reference to variable declaration
      Ref_Type    : Type_Reference;  --  Type of the variable
   end record;

   --  Member access expression (obj.field)
   type Member_Expr is record
      Base        : Semantic_Node (Kind_Member_Expr);
      Object      : Node_ID;  --  Reference to object expression
      Member_Name : IR_Name;
      Member_Binding : Node_ID;  --  Reference to member declaration
      Member_Type : Type_Reference;  --  Type of the member
   end record;

   --  Array access expression (arr[i])
   type Array_Access_Expr is record
      Base        : Semantic_Node (Kind_Array_Access);
      Array_Expr  : Node_ID;  --  Reference to array expression
      Index_Expr  : Node_ID;  --  Reference to index expression
      Element_Type : Type_Reference;  --  Type of array element
   end record;

   --  =========================================================================
   --  Operator Expressions
   --  =========================================================================

   --  Binary expression (a + b, a < b, etc.)
   type Binary_Expr is record
      Base        : Semantic_Node (Kind_Binary_Expr);
      Op          : Binary_Operator;
      Left        : Node_ID;  --  Reference to left operand
      Right       : Node_ID;  --  Reference to right operand
      Result_Type : Type_Reference;  --  Type of result
   end record;

   --  Unary expression (-a, !b, etc.)
   type Unary_Expr is record
      Base        : Semantic_Node (Kind_Unary_Expr);
      Op          : Unary_Operator;
      Operand     : Node_ID;  --  Reference to operand
      Result_Type : Type_Reference;  --  Type of result
   end record;

   --  Ternary expression (a ? b : c)
   type Ternary_Expr is record
      Base        : Semantic_Node (Kind_Ternary_Expr);
      Condition   : Node_ID;  --  Reference to condition
      Then_Expr   : Node_ID;  --  Reference to then expression
      Else_Expr   : Node_ID;  --  Reference to else expression
      Result_Type : Type_Reference;  --  Type of result
   end record;

   --  Cast expression ((Type)a)
   type Cast_Expr is record
      Base        : Semantic_Node (Kind_Cast_Expr);
      Expr        : Node_ID;  --  Reference to expression to cast
      Target_Type : Type_Reference;  --  Target type
   end record;

   --  =========================================================================
   --  Call Expressions
   --  =========================================================================

   --  Maximum arguments per call (reduced for stack usage)
   Max_Call_Args : constant := 8;

   type Arg_List is array (1 .. Max_Call_Args) of Node_ID;

   --  Function call expression
   type Function_Call_Expr is record
      Base          : Semantic_Node (Kind_Function_Call);
      Func_Binding  : Node_ID;  --  Reference to function declaration
      Arg_Count     : Natural range 0 .. Max_Call_Args := 0;
      Arguments     : Arg_List;
      Result_Type   : Type_Reference;  --  Return type
      Is_Intrinsic  : Boolean := False;  --  Is this an intrinsic call?
   end record;

   --  =========================================================================
   --  Aggregate Expressions
   --  =========================================================================

   --  Maximum elements per aggregate (reduced for stack usage)
   Max_Aggregate_Elems : constant := 16;

   type Elem_List is array (1 .. Max_Aggregate_Elems) of Node_ID;

   --  Array initialization expression ([a, b, c])
   type Array_Init_Expr is record
      Base        : Semantic_Node (Kind_Array_Init);
      Elem_Count  : Natural range 0 .. Max_Aggregate_Elems := 0;
      Elements    : Elem_List;
      Array_Type  : Type_Reference;  --  Type of the array
   end record;

   --  Struct initialization expression ({field: value, ...})
   type Struct_Init_Expr is record
      Base        : Semantic_Node (Kind_Struct_Init);
      Field_Count : Natural range 0 .. Max_Aggregate_Elems := 0;
      Fields      : Elem_List;  --  References to field initializers
      Struct_Type : Type_Reference;  --  Type of the struct
   end record;

   --  =========================================================================
   --  Validation Functions
   --  =========================================================================

   --  Check if an integer literal is valid
   function Is_Valid_Integer_Literal (L : Integer_Literal_Expr) return Boolean
      with Post => (if Is_Valid_Integer_Literal'Result then
                    Is_Valid_Semantic_Node (L.Base) and then
                    Is_Primitive_Type (L.Lit_Type));

   --  Check if a float literal is valid
   function Is_Valid_Float_Literal (L : Float_Literal_Expr) return Boolean
      with Post => (if Is_Valid_Float_Literal'Result then
                    Is_Valid_Semantic_Node (L.Base));

   --  Check if a string literal is valid
   function Is_Valid_String_Literal (L : String_Literal_Expr) return Boolean
      with Post => (if Is_Valid_String_Literal'Result then
                    Is_Valid_Semantic_Node (L.Base) and then
                    Name_Strings.Length (L.Value) > 0);

   --  Check if a boolean literal is valid
   function Is_Valid_Bool_Literal (L : Bool_Literal_Expr) return Boolean
      with Post => (if Is_Valid_Bool_Literal'Result then
                    Is_Valid_Semantic_Node (L.Base));

   --  Check if a variable reference is valid
   function Is_Valid_Var_Ref (V : Var_Ref_Expr) return Boolean
      with Post => (if Is_Valid_Var_Ref'Result then
                    Is_Valid_Semantic_Node (V.Base) and then
                    Name_Strings.Length (V.Var_Name) > 0 and then
                    Is_Valid_Node_ID (V.Var_Binding));

   --  Check if a binary expression is valid
   function Is_Valid_Binary_Expr (B : Binary_Expr) return Boolean
      with Post => (if Is_Valid_Binary_Expr'Result then
                    Is_Valid_Semantic_Node (B.Base) and then
                    Is_Valid_Node_ID (B.Left) and then
                    Is_Valid_Node_ID (B.Right));

   --  Check if a unary expression is valid
   function Is_Valid_Unary_Expr (U : Unary_Expr) return Boolean
      with Post => (if Is_Valid_Unary_Expr'Result then
                    Is_Valid_Semantic_Node (U.Base) and then
                    Is_Valid_Node_ID (U.Operand));

   --  Check if a function call is valid
   function Is_Valid_Function_Call (F : Function_Call_Expr) return Boolean
      with Post => (if Is_Valid_Function_Call'Result then
                    Is_Valid_Semantic_Node (F.Base) and then
                    Is_Valid_Node_ID (F.Func_Binding));

end Semantic_IR.Expressions;