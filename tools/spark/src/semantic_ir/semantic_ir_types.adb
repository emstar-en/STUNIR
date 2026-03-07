-------------------------------------------------------------------------------
--  STUNIR Semantic IR Types Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of validation functions for Semantic IR types.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_IR.Types is

   --  Check if a node ID is valid (non-empty and reasonable length)
   function Is_Valid_Node_ID (ID : Node_ID) return Boolean is
      Len : constant Natural := Name_Strings.Length (ID);
   begin
      --  Node IDs must have at least 3 characters (e.g., "n_0")
      --  and must start with valid prefix
      if Len < 3 then
         return False;
      end if;

      --  Check for valid prefix (n_ for nodes, f_ for functions, etc.)
      declare
         Prefix : constant String := Name_Strings.To_String (ID)(1 .. 2);
      begin
         return Prefix = "n_" or Prefix = "f_" or
                Prefix = "t_" or Prefix = "v_" or
                Prefix = "c_" or Prefix = "m_";
      end;
   end Is_Valid_Node_ID;

   --  Check if a hash is valid (correct length for sha256: prefix + hex)
   function Is_Valid_Hash (H : IR_Hash) return Boolean is
      Len : constant Natural := Hash_Strings.Length (H);
   begin
      --  Hash must be exactly 71 characters: "sha256:" (7) + 64 hex chars
      if Len /= 71 then
         return False;
      end if;

      --  Check for sha256: prefix
      declare
         Prefix : constant String := Hash_Strings.To_String (H)(1 .. 7);
      begin
         if Prefix /= "sha256:" then
            return False;
         end if;
      end;

      --  Check that remaining characters are valid hex
      declare
         Hash_Str : constant String := Hash_Strings.To_String (H);
         C : Character;
      begin
         for I in 8 .. 71 loop
            C := Hash_Str (I);
            if not (C in '0' .. '9' | 'a' .. 'f' | 'A' .. 'F') then
               return False;
            end if;
         end loop;
      end;

      return True;
   end Is_Valid_Hash;

end Semantic_IR.Types;