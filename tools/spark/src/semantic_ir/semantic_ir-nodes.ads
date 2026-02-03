-- STUNIR Semantic IR Nodes Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;

package Semantic_IR.Nodes with
   SPARK_Mode => On
is
   -- Base node type (discriminated record)
   type IR_Node (Kind : IR_Node_Kind) is record
      Node_ID  : Node_ID;
      Location : Source_Location;
      Hash     : IR_Hash;
      
      case Kind is
         when Kind_Integer_Literal =>
            Int_Value : Long_Long_Integer;
            Int_Radix : Integer range 2 .. 16 := 10;
            
         when Kind_Float_Literal =>
            Float_Value : Long_Float;
            
         when Kind_String_Literal =>
            Str_Value : IR_Name;
            
         when Kind_Bool_Literal =>
            Bool_Value : Boolean;
            
         when Kind_Var_Ref =>
            Var_Name : IR_Name;
            Var_Binding : Node_ID;
            
         when others =>
            null;
      end case;
   end record;
   
   -- Type reference (simplified)
   type Type_Kind is (
      TK_Primitive,
      TK_Array,
      TK_Pointer,
      TK_Struct,
      TK_Function,
      TK_Ref
   );
   
   type Type_Reference (Kind : Type_Kind := TK_Primitive) is record
      case Kind is
         when TK_Primitive =>
            Prim_Type : IR_Primitive_Type;
         when TK_Ref =>
            Type_Name : IR_Name;
            Type_Binding : Node_ID;
         when others =>
            null; -- Simplified for now
      end case;
   end record;
   
   -- Node validation
   function Is_Valid_Node_ID (ID : Node_ID) return Boolean
      with Post => (if Is_Valid_Node_ID'Result then
                       Name_Strings.Length (ID) > 2);
   
   function Is_Valid_Hash (H : IR_Hash) return Boolean
      with Post => (if Is_Valid_Hash'Result then
                       Hash_Strings.Length (H) = 71); -- "sha256:" + 64 hex
   
   function Is_Literal_Kind (Kind : IR_Node_Kind) return Boolean is
      (Kind in Kind_Integer_Literal | Kind_Float_Literal |
               Kind_String_Literal | Kind_Bool_Literal);
   
end Semantic_IR.Nodes;
