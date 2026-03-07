-------------------------------------------------------------------------------
--  STUNIR Semantic IR Nodes Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of validation functions for Semantic IR nodes.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Nodes is

   --  Check if a semantic node is valid
   function Is_Valid_Semantic_Node (N : Semantic_Node) return Boolean is
   begin
      --  Node must have valid ID
      if not Is_Valid_Node_ID (N.ID) then
         return False;
      end if;

      --  Node must have valid type reference
      if not Is_Valid_Type_Reference (N.Node_Type) then
         return False;
      end if;

      --  Edge count must be within bounds
      if N.Edge_Count > Max_CFG_Edges then
         return False;
      end if;

      --  Validate edges if present
      for I in 1 .. N.Edge_Count loop
         if not Is_Valid_Node_ID (N.Edges (I).Target_Node) then
            return False;
         end if;
      end loop;

      return True;
   end Is_Valid_Semantic_Node;

   --  Check if a type reference is valid
   function Is_Valid_Type_Reference (T : Type_Reference) return Boolean is
   begin
      case T.Kind is
         when TK_Primitive =>
            --  Primitive types are always valid
            return True;

         when TK_Ref =>
            --  Named type must have non-empty name
            return Name_Strings.Length (T.Type_Name) > 0;

         when TK_Array =>
            --  Array type must have valid element type
            return Is_Valid_Type_Reference (T.Element_Type);

         when TK_Pointer =>
            --  Pointer type must have valid pointed type
            return Is_Valid_Type_Reference (T.Pointed_Type);

         when TK_Function =>
            --  Function type must have valid return type
            return Is_Valid_Type_Reference (T.Return_Type);

         when TK_Struct =>
            --  Struct type must have non-empty name
            return Name_Strings.Length (T.Struct_Name) > 0;

      end case;
   end Is_Valid_Type_Reference;

end Semantic_IR.Nodes;