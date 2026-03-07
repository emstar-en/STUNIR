--  IR Canonicalization Utilities (dCBOR profile)
--  Shared helper for IR canonicalization in SPARK pipeline.
--
--  Enforces normal_form rules from tools/spark/schema/stunir_ir_v1.dcbor.json:
--    - Field ordering: lexicographic (UTF-8 byte order)
--    - No floats (hard reject)
--    - No duplicate keys (hard reject)
--    - NFC string normalization
--    - Array ordering: types/functions alphabetically by name
--
--  Models MUST NOT invent their own formats.

pragma SPARK_Mode (Off);

with Ada.Strings.Unbounded;

package IR_Canonicalize_DCBOR_Utils is

   use Ada.Strings.Unbounded;

   --  Array type for sorting
   Max_Keys : constant := 256;
   type Unbounded_String_Array is array (Natural range <>) of Unbounded_String;

   --  Canonicalization result with diagnostics
   type Canonicalize_Result is record
      Success      : Boolean;
      Output       : Unbounded_String;
      Error_Msg    : Unbounded_String;
      Warnings     : Natural := 0;
      Keys_Sorted  : Natural := 0;
      Floats_Rejected : Natural := 0;
   end record;

   --  Main canonicalization function (backward compatible)
   function Canonicalize (Input : String) return String;

   --  Full canonicalization with diagnostics
   function Canonicalize_Full (Input : String) return Canonicalize_Result;

   --  Validation-only mode (returns input unchanged if valid)
   function Validate_Only (Input : String) return Canonicalize_Result;

end IR_Canonicalize_DCBOR_Utils;
