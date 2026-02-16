-------------------------------------------------------------------------------
--  STUNIR Receipt Types - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides receipt-related types for build tracking.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;
with Epoch_Types;    use Epoch_Types;

package Receipt_Types is

   --  Maximum receipts
   Max_Receipts : constant := 256;
   Max_Input_Files : constant := 64;

   --  Receipt schema
   Receipt_Schema_V1 : constant String := "stunir.receipt.v1";

   --  Receipt status
   type Receipt_Status is (
      Receipt_Created,
      Receipt_Skipped_No_Compiler,
      Receipt_Skipped_No_Source,
      Receipt_Binary_Emitted,
      Receipt_Compilation_Failed,
      Receipt_Verification_Passed,
      Receipt_Verification_Failed
   );

   --  Receipt kind
   type Receipt_Kind is (
      Kind_Build,        --  Build artifact receipt
      Kind_Verification, --  Verification receipt
      Kind_Provenance,   --  Provenance receipt
      Kind_Manifest      --  Manifest receipt
   );

   --  Input file entry
   type Input_File_Entry is record
      Path   : Path_String := Empty_Path;
      Hash   : Hash_Hex := Zero_Hash;
   end record;

   type Input_File_Array is array (Positive range <>) of Input_File_Entry;
   subtype Input_File_Vector is Input_File_Array (1 .. Max_Input_Files);

   --  Tool info (for recording tool used)
   type Tool_Info is record
      Path    : Path_String := Empty_Path;
      Hash    : Hash_Hex := Zero_Hash;
      Version : Medium_String := Empty_Medium;
   end record;

   --  Build receipt
   type Build_Receipt is record
      Schema       : Short_String := Empty_Short;
      Kind         : Receipt_Kind := Kind_Build;
      Target       : Path_String := Empty_Path;
      Target_Hash  : Hash_Hex := Zero_Hash;
      Status       : Receipt_Status := Receipt_Created;
      Epoch        : Epoch_Value := 0;
      Tool         : Tool_Info;
      Inputs       : Input_File_Vector := (others => (others => <>));
      Input_Count  : Natural := 0;
      Is_Valid     : Boolean := False;
   end record;

   --  Empty receipt
   Empty_Receipt : constant Build_Receipt := (
      Schema       => Empty_Short,
      Kind         => Kind_Build,
      Target       => Empty_Path,
      Target_Hash  => Zero_Hash,
      Status       => Receipt_Created,
      Epoch        => 0,
      Tool         => (others => <>),
      Inputs       => (others => (others => <>)),
      Input_Count  => 0,
      Is_Valid     => False
   );

   --  Receipt array
   type Receipt_Array is array (Positive range <>) of Build_Receipt;
   subtype Receipt_Vector is Receipt_Array (1 .. Max_Receipts);

   --  Receipt registry
   type Receipt_Registry is record
      Receipts : Receipt_Vector := (others => Empty_Receipt);
      Count    : Natural := 0;
   end record;

   --  Status to string
   function Status_To_String (S : Receipt_Status) return String;

   --  Kind to string
   function Kind_To_String (K : Receipt_Kind) return String;

   --  Check if receipt is valid
   function Is_Valid_Receipt (R : Build_Receipt) return Boolean is
     (R.Is_Valid and R.Target.Length > 0);

end Receipt_Types;
