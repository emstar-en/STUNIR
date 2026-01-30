-------------------------------------------------------------------------------
--  STUNIR Receipt Generator - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides receipt generation functionality.
--  Migrated from: tools/record_receipt.py
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings;  use Stunir_Strings;
with Stunir_Hashes;   use Stunir_Hashes;
with Epoch_Types;     use Epoch_Types;
with Receipt_Types;   use Receipt_Types;

package Receipt_Generator is

   --  Initialize a new receipt
   procedure Initialize_Receipt (
      Receipt : out Build_Receipt;
      Kind    : Receipt_Kind := Kind_Build)
     with
       Post => not Receipt.Is_Valid;

   --  Set receipt target
   procedure Set_Target (
      Receipt     : in out Build_Receipt;
      Target_Path : Path_String)
     with
       Pre  => Target_Path.Length > 0,
       Post => Receipt.Target.Length > 0;

   --  Set receipt epoch
   procedure Set_Epoch (
      Receipt : in out Build_Receipt;
      Epoch   : Epoch_Value);

   --  Set receipt status
   procedure Set_Status (
      Receipt : in out Build_Receipt;
      Status  : Receipt_Status);

   --  Set tool info
   procedure Set_Tool (
      Receipt   : in out Build_Receipt;
      Tool_Path : Path_String;
      Tool_Hash : Hash_Hex);

   --  Add input file to receipt
   procedure Add_Input (
      Receipt : in out Build_Receipt;
      Path    : Path_String;
      Hash    : Hash_Hex;
      Success : out Boolean)
     with
       Pre  => Path.Length > 0 and then Receipt.Input_Count < Max_Input_Files;

   --  Finalize and validate receipt
   procedure Finalize_Receipt (
      Receipt : in out Build_Receipt)
     with
       Post => Receipt.Is_Valid = (Receipt.Target.Length > 0);

   --  Compute receipt core ID (for deterministic identification)
   function Compute_Receipt_Core_Id (
      Receipt : Build_Receipt) return Hash_Hex
     with
       Pre => Receipt.Is_Valid;

   --  Generate receipt from compilation result
   procedure Generate_Compilation_Receipt (
      Target_Path   : Path_String;
      Tool_Path     : Path_String;
      Epoch         : Epoch_Value;
      Compiled_Ok   : Boolean;
      Receipt       : out Build_Receipt)
     with
       Pre  => Target_Path.Length > 0,
       Post => Receipt.Is_Valid;

   --  Add receipt to registry
   procedure Add_To_Registry (
      Registry : in out Receipt_Registry;
      Receipt  : Build_Receipt;
      Success  : out Boolean)
     with
       Pre  => Receipt.Is_Valid and then Registry.Count < Max_Receipts;

   --  Find receipt by target path
   function Find_Receipt (
      Registry : Receipt_Registry;
      Target   : Path_String) return Natural;

end Receipt_Generator;
