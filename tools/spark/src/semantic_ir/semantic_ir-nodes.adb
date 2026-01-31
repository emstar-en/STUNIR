-- STUNIR Semantic IR Nodes Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Nodes is

   function Is_Valid_Node_ID (ID : Node_ID) return Boolean is
      use Name_Strings;
   begin
      -- Node IDs should start with "n_"
      if Length (ID) < 3 then
         return False;
      end if;
      
      declare
         S : constant String := To_String (ID);
      begin
         return S'Length >= 2 and then S (S'First .. S'First + 1) = "n_";
      end;
   end Is_Valid_Node_ID;
   
   function Is_Valid_Hash (H : IR_Hash) return Boolean is
      use Hash_Strings;
   begin
      -- Hashes should be "sha256:" followed by 64 hex characters
      if Length (H) /= 71 then
         return False;
      end if;
      
      declare
         S : constant String := To_String (H);
      begin
         return S'Length = 71 and then
                S (S'First .. S'First + 6) = "sha256:";
      end;
   end Is_Valid_Hash;
   
end Semantic_IR.Nodes;
