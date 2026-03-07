-------------------------------------------------------------------------------
--  STUNIR Semantic IR Nodes Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines the core AST node types for the STUNIR Semantic IR.
--  It provides discriminated records for different node kinds with semantic
--  annotations including type bindings, safety levels, and target categories.
--
--  Key differences from flat IR nodes:
--  - Explicit type bindings (Type_Reference with binding to declaration)
--  - Safety level annotations for DO-178C compliance
--  - Target category annotations for multi-target code generation
--  - Control flow graph edges for CFG representation
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
--
--  Normal Form: All nodes must conform to the normal_form rules defined in
--               tools/spark/schema/stunir_ir_v1.dcbor.json
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;

package Semantic_IR.Nodes with
   SPARK_Mode => On
is
   --  =========================================================================
   --  Source Location Tracking
   --  =========================================================================

   --  Source location information for error reporting and debugging
   type Source_Location is record
      Line   : Natural := 0;    --  Line number (1-based, 0 = unknown)
      Column : Natural := 0;    --  Column number (1-based, 0 = unknown)
      Length : Natural := 0;    --  Token length (0 = unknown)
      File   : IR_Path;         --  Source file path
   end record;

   --  =========================================================================
   --  Type References (with explicit bindings)
   --  =========================================================================

   --  Type reference with discriminant for different type kinds
   --  Unlike flat IR, Semantic IR includes explicit bindings to declarations
   type Type_Reference (Kind : Type_Kind := TK_Primitive) is record
      case Kind is
         --  Primitive type: specific primitive
         when TK_Primitive =>
            Prim_Type : IR_Primitive_Type;

         --  Named type reference: name and binding
         when TK_Ref =>
            Type_Name    : IR_Name;   --  Type name
            Type_Binding : Node_ID;   --  Reference to type declaration

         --  Array type: element type and size
         when TK_Array =>
            Element_Type : Type_Reference;  --  Element type (nested)
            Array_Size   : Natural := 0;    --  0 = dynamic/unknown

         --  Pointer type: pointed type
         when TK_Pointer =>
            Pointed_Type : Type_Reference;  --  Pointed type (nested)

         --  Function type: return and parameter types
         when TK_Function =>
            Return_Type  : Type_Reference;  --  Return type (nested)
            --  Parameters handled separately due to SPARK constraints

         --  Struct type: name and binding
         when TK_Struct =>
            Struct_Name    : IR_Name;   --  Struct name
            Struct_Binding : Node_ID;   --  Reference to struct declaration

      end case;
   end record;

   --  =========================================================================
   --  Semantic Annotations
   --  =========================================================================

   --  Semantic annotations attached to nodes for code generation
   type Semantic_Annotations is record
      Safety_Level     : Safety_Level := Level_None;
      Target_Category   : Target_Category := Target_Native;
      Inline_Hint       : Inline_Hint := Inline_None;
      Is_Volatile       : Boolean := False;
      Is_Atomic         : Boolean := False;
      Is_Restricted     : Boolean := False;
   end record;

   --  =========================================================================
   --  Control Flow Graph Edge
   --  =========================================================================

   --  Edge in the control flow graph
   type CFG_Edge is record
      Target_Node : Node_ID;       --  Target node ID
      Edge_Kind   : CFG_Edge_Kind; --  Kind of edge
   end record;

   --  Bounded list of CFG edges
   Max_CFG_Edges : constant := 4;  --  Most nodes have <= 4 edges
   type CFG_Edge_List is array (1 .. Max_CFG_Edges) of CFG_Edge;

   --  =========================================================================
   --  Semantic IR AST Node
   --  =========================================================================

   --  Base AST node type with discriminant for variant-specific data.
   --  Each node has a unique ID, source location, content hash,
   --  type reference, and semantic annotations.
   type Semantic_Node (Kind : Semantic_Node_Kind) is record
      ID            : Node_ID;            --  Unique identifier for this node
      Location      : Source_Location;    --  Source location for debugging
      Hash          : IR_Hash;           --  Content hash for verification
      Node_Type     : Type_Reference;    --  Type of this node (with binding)
      Annotations   : Semantic_Annotations; --  Semantic annotations

      --  Control flow edges (for CFG representation)
      Edge_Count    : Natural range 0 .. Max_CFG_Edges := 0;
      Edges         : CFG_Edge_List;

      --  Variant-specific fields based on node kind
      case Kind is
         --  Integer literal: value and radix
         when Kind_Integer_Literal =>
            Int_Value : Long_Long_Integer;
            Int_Radix : Integer range 2 .. 16 := 10;

         --  Float literal: value
         when Kind_Float_Literal =>
            Float_Value : Long_Float;

         --  String literal: value as bounded string
         when Kind_String_Literal =>
            Str_Value : IR_Name;

         --  Boolean literal: value
         when Kind_Bool_Literal =>
            Bool_Value : Boolean;

         --  Variable reference: name and binding
         when Kind_Var_Ref =>
            Var_Name    : IR_Name;   --  Variable name
            Var_Binding : Node_ID;   --  Reference to declaration

         --  Binary expression: operator and operands
         when Kind_Binary_Expr =>
            Binary_Op   : Binary_Operator;
            Left_Op     : Node_ID;   --  Left operand
            Right_Op    : Node_ID;   --  Right operand

         --  Unary expression: operator and operand
         when Kind_Unary_Expr =>
            Unary_Op    : Unary_Operator;
            Operand     : Node_ID;   --  Operand

         --  Function call: function reference and arguments
         when Kind_Function_Call =>
            Func_Binding : Node_ID;  --  Reference to function declaration
            --  Arguments handled separately due to SPARK constraints

         --  Member access: object and member name
         when Kind_Member_Expr =>
            Object_Node  : Node_ID;  --  Object expression
            Member_Name  : IR_Name;  --  Member name

         --  Array access: array and index
         when Kind_Array_Access =>
            Array_Node   : Node_ID;  --  Array expression
            Index_Node   : Node_ID;  --  Index expression

         --  Cast expression: expression and target type
         when Kind_Cast_Expr =>
            Cast_Expr    : Node_ID;  --  Expression to cast
            Target_Type  : Type_Reference;  --  Target type

         --  Ternary expression: condition, then, else
         when Kind_Ternary_Expr =>
            Cond_Node    : Node_ID;  --  Condition
            Then_Node    : Node_ID;  --  Then branch
            Else_Node    : Node_ID;  --  Else branch

         --  Other node kinds have no additional fields in base record
         when others =>
            null;
      end case;
   end record;

   --  =========================================================================
   --  Validation Functions
   --  =========================================================================

   --  Check if a semantic node is valid
   function Is_Valid_Semantic_Node (N : Semantic_Node) return Boolean
      with Post => (if Is_Valid_Semantic_Node'Result then
                    Is_Valid_Node_ID (N.ID));

   --  Check if a type reference is valid
   function Is_Valid_Type_Reference (T : Type_Reference) return Boolean;

   --  Check if a source location is valid (has file and line)
   function Is_Valid_Source_Location (L : Source_Location) return Boolean
      is (Name_Strings.Length (L.File) > 0 and then L.Line > 0);

   --  Get the type kind from a type reference
   function Get_Type_Kind (T : Type_Reference) return Type_Kind
      is (T.Kind);

   --  Check if a type is a primitive type
   function Is_Primitive_Type (T : Type_Reference) return Boolean
      is (T.Kind = TK_Primitive);

   --  Check if a type is a named reference
   function Is_Named_Type (T : Type_Reference) return Boolean
      is (T.Kind = TK_Ref);

end Semantic_IR.Nodes;