-------------------------------------------------------------------------------
--  STUNIR Semantic IR Expressions Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of validation functions for Semantic IR expressions.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Expressions is

   --  Check if an integer literal is valid
   function Is_Valid_Integer_Literal (L : Integer_Literal_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (L.Base) then
         return False;
      end if;

      --  Type must be primitive (integer)
      if not Is_Primitive_Type (L.Lit_Type) then
         return False;
      end if;

      --  Radix must be valid
      if L.Radix < 2 or L.Radix > 16 then
         return False;
      end if;

      return True;
   end Is_Valid_Integer_Literal;

   --  Check if a float literal is valid
   function Is_Valid_Float_Literal (L : Float_Literal_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (L.Base) then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (L.Lit_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Float_Literal;

   --  Check if a string literal is valid
   function Is_Valid_String_Literal (L : String_Literal_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (L.Base) then
         return False;
      end if;

      --  Value must be non-empty
      if Name_Strings.Length (L.Value) = 0 then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (L.Lit_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_String_Literal;

   --  Check if a boolean literal is valid
   function Is_Valid_Bool_Literal (L : Bool_Literal_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (L.Base) then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (L.Lit_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Bool_Literal;

   --  Check if a variable reference is valid
   function Is_Valid_Var_Ref (V : Var_Ref_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (V.Base) then
         return False;
      end if;

      --  Variable name must be non-empty
      if Name_Strings.Length (V.Var_Name) = 0 then
         return False;
      end if;

      --  Binding must be valid
      if not Is_Valid_Node_ID (V.Var_Binding) then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (V.Ref_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Var_Ref;

   --  Check if a binary expression is valid
   function Is_Valid_Binary_Expr (B : Binary_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (B.Base) then
         return False;
      end if;

      --  Left operand must be valid
      if not Is_Valid_Node_ID (B.Left) then
         return False;
      end if;

      --  Right operand must be valid
      if not Is_Valid_Node_ID (B.Right) then
         return False;
      end if;

      --  Result type must be valid
      if not Is_Valid_Type_Reference (B.Result_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Binary_Expr;

   --  Check if a unary expression is valid
   function Is_Valid_Unary_Expr (U : Unary_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (U.Base) then
         return False;
      end if;

      --  Operand must be valid
      if not Is_Valid_Node_ID (U.Operand) then
         return False;
      end if;

      --  Result type must be valid
      if not Is_Valid_Type_Reference (U.Result_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Unary_Expr;

   --  Check if a function call is valid
   function Is_Valid_Function_Call (F : Function_Call_Expr) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (F.Base) then
         return False;
      end if;

      --  Function binding must be valid
      if not Is_Valid_Node_ID (F.Func_Binding) then
         return False;
      end if;

      --  Argument count must be within bounds
      if F.Arg_Count > Max_Call_Args then
         return False;
      end if;

      --  Validate all arguments
      for I in 1 .. F.Arg_Count loop
         if not Is_Valid_Node_ID (F.Arguments (I)) then
            return False;
         end if;
      end loop;

      --  Result type must be valid
      if not Is_Valid_Type_Reference (F.Result_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Function_Call;

end Semantic_IR.Expressions;