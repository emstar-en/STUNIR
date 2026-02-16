-------------------------------------------------------------------------------
--  STUNIR Epoch Types - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides epoch-related types for deterministic builds.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;

package Epoch_Types is

   --  Epoch value (Unix timestamp)
   type Epoch_Value is range 0 .. 2**63 - 1;
   Zero_Epoch_Value : constant Epoch_Value := 0;
   
   --  Epoch source enumeration
   type Epoch_Source is (
      Source_Unknown,
      Source_Env_Build_Epoch,      --  STUNIR_BUILD_EPOCH
      Source_Env_Source_Date,      --  SOURCE_DATE_EPOCH
      Source_Derived_Spec_Digest,  --  Derived from spec/ tree digest
      Source_Git_Commit,           --  From git log
      Source_Zero,                 --  Fallback to 0
      Source_Current_Time          --  Non-deterministic (current time)
   );

   --  Epoch selection result
   type Epoch_Selection is record
      Value           : Epoch_Value := Zero_Epoch_Value;
      Source          : Epoch_Source := Source_Unknown;
      Is_Deterministic : Boolean := True;
      Spec_Digest     : Hash_Hex := Zero_Hash;
   end record;

   --  Default (zero) selection
   Default_Epoch_Selection : constant Epoch_Selection := (
      Value           => Zero_Epoch_Value,
      Source          => Source_Zero,
      Is_Deterministic => True,
      Spec_Digest     => Zero_Hash
   );

   --  Epoch JSON record (for serialization)
   type Epoch_JSON is record
      Selected_Epoch   : Epoch_Value := 0;
      Epoch_Source_Str : Medium_String := Empty_Medium;
      Deterministic    : Boolean := True;
      Spec_Digest_Hex  : Hash_Hex := Zero_Hash;
   end record;

   --  Convert source to string representation
   function Source_To_String (S : Epoch_Source) return String;

   --  Convert string to source (parsing)
   function String_To_Source (S : String) return Epoch_Source;

   --  Check if epoch is deterministic
   function Is_Deterministic_Source (S : Epoch_Source) return Boolean is
     (S /= Source_Current_Time and S /= Source_Unknown);

end Epoch_Types;
