-------------------------------------------------------------------------------
--  STUNIR Hash Utilities - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
--
--  NOTE: This is a simplified implementation for SPARK compatibility.
--  Production use should integrate with a proper cryptographic library.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Stunir_Hashes is

   --  Hex digit lookup
   Hex_Digits : constant String := "0123456789abcdef";

   -------------------------------------------------------------------------
   --  Hashes_Equal: Compare two hashes
   -------------------------------------------------------------------------
   function Hashes_Equal (Left, Right : Hash_Hex) return Boolean is
   begin
      return Left.Data = Right.Data;
   end Hashes_Equal;

   -------------------------------------------------------------------------
   --  To_String: Convert hash to string
   -------------------------------------------------------------------------
   function To_String (H : Hash_Hex) return String is
   begin
      return H.Data;
   end To_String;

   -------------------------------------------------------------------------
   --  From_String: Create hash from string
   -------------------------------------------------------------------------
   function From_String (S : String) return Hash_Hex is
      Result : Hash_Hex;
   begin
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end From_String;

   -------------------------------------------------------------------------
   --  Is_Valid_Hex: Check if string is valid hex
   -------------------------------------------------------------------------
   function Is_Valid_Hex (S : String) return Boolean is
   begin
      for I in S'Range loop
         declare
            C : constant Character := S (I);
         begin
            if not (C in '0' .. '9' | 'a' .. 'f' | 'A' .. 'F') then
               return False;
            end if;
         end;
      end loop;
      return True;
   end Is_Valid_Hex;

   -------------------------------------------------------------------------
   --  SHA256_Init: Initialize SHA-256 context
   -------------------------------------------------------------------------
   procedure SHA256_Init (State : out SHA256_State) is
   begin
      State := (Accumulator => 0, Count => 0, Finalized => False);
   end SHA256_Init;

   -------------------------------------------------------------------------
   --  SHA256_Update: Update SHA-256 with data
   --  NOTE: This is a simplified hash for demonstration.
   --  Real implementation would use proper SHA-256 algorithm.
   -------------------------------------------------------------------------
   procedure SHA256_Update (
      State : in out SHA256_State;
      Byte  : Character)
   is
   begin
      if not State.Finalized then
         --  Simple hash accumulation (NOT cryptographically secure)
         State.Accumulator := (State.Accumulator * 31 + Character'Pos (Byte)) mod (2 ** 28);
         State.Count := State.Count + 1;
      end if;
   end SHA256_Update;

   -------------------------------------------------------------------------
   --  SHA256_Final: Finalize SHA-256 and get hash
   --  NOTE: This generates a deterministic output based on accumulated state.
   --  Real implementation would complete the SHA-256 algorithm.
   -------------------------------------------------------------------------
   procedure SHA256_Final (
      State : in out SHA256_State;
      Hash  : out Hash_Hex)
   is
      Value : Natural := State.Accumulator;
      Temp  : Natural;
   begin
      State.Finalized := True;

      --  Generate a deterministic 64-character hex string
      --  This is a placeholder - real SHA-256 would produce proper hash
      for I in reverse 1 .. Hash_Length loop
         Temp := Value mod 16;
         Hash.Data (I) := Hex_Digits (Temp + 1);
         Value := Value / 16;
         if Value = 0 and I > 1 then
            --  Mix in count for variety
            Value := (State.Count * 7 + I * 13) mod (2 ** 28);
         end if;
      end loop;
   end SHA256_Final;

   -------------------------------------------------------------------------
   --  Hash_String: Simple hash of a string
   -------------------------------------------------------------------------
   function Hash_String (S : String) return Hash_Hex is
      State : SHA256_State;
      Result : Hash_Hex;
   begin
      SHA256_Init (State);
      for I in S'Range loop
         SHA256_Update (State, S (I));
      end loop;
      SHA256_Final (State, Result);
      return Result;
   end Hash_String;

end Stunir_Hashes;
