-------------------------------------------------------------------------------
--  STUNIR Epoch Selector - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides deterministic epoch selection logic.
--  Migrated from: tools/epoch.py
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;
with Epoch_Types;    use Epoch_Types;

package Epoch_Selector is

   --  Maximum environment variable value length
   Max_Env_Value_Length : constant := 256;

   --  Environment variable container
   subtype Env_Value_Length is Natural range 0 .. Max_Env_Value_Length;
   type Env_Value is record
      Data   : String (1 .. Max_Env_Value_Length) := (others => ' ');
      Length : Env_Value_Length := 0;
      Exists : Boolean := False;
   end record;

   Empty_Env_Value : constant Env_Value := (
      Data   => (others => ' '),
      Length => 0,
      Exists => False
   );

   --  Check if environment variable exists
   function Has_Env_Variable (Name : String) return Boolean;

   --  Get environment variable value
   function Get_Env_Variable (Name : String) return Env_Value
     with Post => (if not Get_Env_Variable'Result.Exists then 
                      Get_Env_Variable'Result.Length = 0);

   --  Parse epoch value from string
   function Parse_Epoch_Value (S : String) return Epoch_Value
     with Pre => S'Length > 0 and S'Length <= 20;

   --  Compute spec directory digest (simplified - returns hash)
   function Compute_Spec_Digest (Spec_Root : Path_String) return Hash_Hex;

   --  Derive epoch from spec digest (first 8 hex chars as integer)
   function Derive_Epoch_From_Digest (Digest : Hash_Hex) return Epoch_Value;

   --  Main epoch selection procedure
   --  Priority: STUNIR_BUILD_EPOCH > SOURCE_DATE_EPOCH > DERIVED > ZERO
   procedure Select_Epoch (
      Spec_Root    : Path_String;
      Allow_Current : Boolean := False;
      Selection    : out Epoch_Selection)
     with
       Post => Selection.Is_Deterministic or Allow_Current;

   --  Convert selection to JSON record
   function To_JSON (Selection : Epoch_Selection) return Epoch_JSON;

   --  Validate epoch selection
   function Is_Valid_Selection (Selection : Epoch_Selection) return Boolean is
     (Selection.Source /= Source_Unknown);

end Epoch_Selector;
