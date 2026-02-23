-------------------------------------------------------------------------------
--  STUNIR IR Nodes Package Specification
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This package defines the core AST node types for the STUNIR IR.
--  It provides discriminated records for different node kinds with formal
--  verification contracts.
--
--  The IR_Node type is the fundamental building block of the STUNIR AST,
--  representing all language constructs in a unified, type-safe manner.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
--
--  REGEX_IR_REF: schema/stunir_regex_ir_v1.dcbor.json
--               group: validation.hash    / pattern_id: sha256_prefixed
--               group: validation.node_id / pattern_id: node_id
--  Is_Valid_Hash  → sha256_prefixed: length check (71 chars = "sha256:" + 64 hex)
--  Is_Valid_Node_ID → node_id: minimum length check (> 2 chars for "n_" + 1)
--  See the regex IR for the full formal definitions and Python equivalents.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with IR.Types; use IR.Types;

package IR.Nodes with
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
   --  IR AST Node
   --  =========================================================================

   --  Base AST node type with discriminant for variant-specific data.
   --  Each node has a unique ID, source location, and content hash.
   type IR_Node (Kind : IR_Node_Kind) is record
      ID       : Node_ID;         --  Unique identifier for this node (renamed from Node_ID to avoid shadowing)
      Location : Source_Location; --  Source location for debugging
      Hash     : IR_Hash;         --  Content hash for verification

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

         --  Other node kinds have no additional fields in base record
         when others =>
            null;
      end case;
   end record;

   --  =========================================================================
   --  Type References
   --  =========================================================================

   --  Kind of type reference for the type system
   type Type_Kind is (
      TK_Primitive,             --  Built-in primitive type
      TK_Array,                 --  Array type
      TK_Pointer,               --  Pointer/reference type
      TK_Struct,                --  Struct/class type
      TK_Function,              --  Function type
      TK_Ref                    --  Named type reference
   );

   --  Type reference with discriminant for different type kinds
   type Type_Reference (Kind : Type_Kind := TK_Primitive) is record
      case Kind is
         --  Primitive type: specific primitive
         when TK_Primitive =>
            Prim_Type : IR_Primitive_Type;

         --  Named type reference: name and binding
         when TK_Ref =>
            Type_Name    : IR_Name;   --  Type name
            Type_Binding : Node_ID;   --  Reference to type declaration

         --  Other type kinds (simplified for now)
         when others =>
            null;
      end case;
   end record;

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
   function Is_Literal_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal | Kind_Float_Literal |
               Kind_String_Literal | Kind_Bool_Literal);

end IR.Nodes;
