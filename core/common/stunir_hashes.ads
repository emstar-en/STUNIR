-------------------------------------------------------------------------------
--  STUNIR Hash Utilities - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides SHA-256 hash computation for SPARK.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package Stunir_Hashes is

   --  SHA-256 produces 32 bytes = 64 hex chars
   Hash_Length : constant := 64;

   --  Hash result type (hex string)
   type Hash_Hex is record
      Data : String (1 .. Hash_Length) := (others => '0');
   end record;

   Zero_Hash : constant Hash_Hex := (Data => (others => '0'));

   --  Compare two hashes
   function Hashes_Equal (Left, Right : Hash_Hex) return Boolean;

   --  Convert hash to string
   function To_String (H : Hash_Hex) return String
     with
       Post => To_String'Result'Length = Hash_Length;

   --  Create hash from string (must be exactly 64 hex chars)
   function From_String (S : String) return Hash_Hex
     with
       Pre => S'Length = Hash_Length;

   --  Validate hex string
   function Is_Valid_Hex (S : String) return Boolean;

   --  SHA-256 state (simplified - real implementation would be more complex)
   type SHA256_State is private;

   --  Initialize SHA-256 context
   procedure SHA256_Init (State : out SHA256_State);

   --  Update SHA-256 with data (simplified - single byte at a time)
   procedure SHA256_Update (
      State : in Out SHA256_State;
      Byte  : Character);

   --  Finalize SHA-256 and get hash
   procedure SHA256_Final (
      State : in Out SHA256_State;
      Hash  : out Hash_Hex);

   --  Simple hash of a string (convenience function)
   function Hash_String (S : String) return Hash_Hex;

private

   --  Simplified SHA-256 state (actual implementation would have full state)
   type SHA256_State is record
      Accumulator : Natural := 0;
      Count       : Natural := 0;
      Finalized   : Boolean := False;
   end record;

end Stunir_Hashes;
