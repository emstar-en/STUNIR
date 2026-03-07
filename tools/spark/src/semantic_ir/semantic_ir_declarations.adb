-------------------------------------------------------------------------------
--  STUNIR Semantic IR Declarations Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of validation functions for Semantic IR declarations.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Declarations is

   --  Check if a function declaration is valid
   function Is_Valid_Function_Decl (F : Function_Decl) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (F.Base) then
         return False;
      end if;

      --  Function name must be non-empty
      if Name_Strings.Length (F.Func_Name) = 0 then
         return False;
      end if;

      --  Parameter count must be within bounds
      if F.Param_Count > Max_Params then
         return False;
      end if;

      --  Return type must be valid
      if not Is_Valid_Type_Reference (F.Return_Type) then
         return False;
      end if;

      --  Validate all parameters
      for I in 1 .. F.Param_Count loop
         if not Is_Valid_Type_Reference (F.Params (I).Param_Type) then
            return False;
         end if;
      end loop;

      return True;
   end Is_Valid_Function_Decl;

   --  Check if a type declaration is valid
   function Is_Valid_Type_Decl (T : Type_Decl) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (T.Base) then
         return False;
      end if;

      --  Type name must be non-empty
      if Name_Strings.Length (T.Type_Name) = 0 then
         return False;
      end if;

      --  Field count must be within bounds
      if T.Field_Count > Max_Fields then
         return False;
      end if;

      --  Validate all fields if struct
      if T.Is_Struct then
         for I in 1 .. T.Field_Count loop
            if not Is_Valid_Type_Reference (T.Fields (I).Field_Type) then
               return False;
            end if;
         end loop;
      end if;

      return True;
   end Is_Valid_Type_Decl;

   --  Check if a constant declaration is valid
   function Is_Valid_Const_Decl (C : Const_Decl) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (C.Base) then
         return False;
      end if;

      --  Constant name must be non-empty
      if Name_Strings.Length (C.Const_Name) = 0 then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (C.Const_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Const_Decl;

   --  Check if a variable declaration is valid
   function Is_Valid_Var_Decl (V : Var_Decl) return Boolean is
   begin
      --  Base node must be valid
      if not Is_Valid_Semantic_Node (V.Base) then
         return False;
      end if;

      --  Variable name must be non-empty
      if Name_Strings.Length (V.Var_Name) = 0 then
         return False;
      end if;

      --  Type must be valid
      if not Is_Valid_Type_Reference (V.Var_Type) then
         return False;
      end if;

      return True;
   end Is_Valid_Var_Decl;

end Semantic_IR.Declarations;