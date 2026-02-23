--  IR Canonicalization Utilities (dCBOR profile)
--  Shared helper for IR canonicalization in SPARK pipeline.

pragma SPARK_Mode (Off);

package IR_Canonicalize_DCBOR_Utils is

   function Canonicalize (Input : String) return String;

end IR_Canonicalize_DCBOR_Utils;
