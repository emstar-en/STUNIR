-------------------------------------------------------------------------------
--  STUNIR Receipt Generator - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Receipt_Generator is

   --  Initialize a new receipt
   procedure Initialize_Receipt (
      Receipt : out Build_Receipt;
      Kind    : Receipt_Kind := Kind_Build)
   is
   begin
      Receipt := Empty_Receipt;
      Receipt.Kind := Kind;
      Receipt.Schema := Make_Short (Receipt_Schema_V1);
   end Initialize_Receipt;

   --  Set receipt target
   procedure Set_Target (
      Receipt     : in out Build_Receipt;
      Target_Path : Path_String)
   is
   begin
      Receipt.Target := Target_Path;
   end Set_Target;

   --  Set receipt epoch
   procedure Set_Epoch (
      Receipt : in out Build_Receipt;
      Epoch   : Epoch_Value)
   is
   begin
      Receipt.Epoch := Epoch;
   end Set_Epoch;

   --  Set receipt status
   procedure Set_Status (
      Receipt : in out Build_Receipt;
      Status  : Receipt_Status)
   is
   begin
      Receipt.Status := Status;
   end Set_Status;

   --  Set tool info
   procedure Set_Tool (
      Receipt   : in out Build_Receipt;
      Tool_Path : Path_String;
      Tool_Hash : Hash_Hex)
   is
   begin
      Receipt.Tool.Path := Tool_Path;
      Receipt.Tool.Hash := Tool_Hash;
   end Set_Tool;

   --  Add input file to receipt
   procedure Add_Input (
      Receipt : in out Build_Receipt;
      Path    : Path_String;
      Hash    : Hash_Hex;
      Success : out Boolean)
   is
   begin
      if Receipt.Input_Count >= Max_Input_Files then
         Success := False;
         return;
      end if;
      
      Receipt.Input_Count := Receipt.Input_Count + 1;
      Receipt.Inputs (Receipt.Input_Count).Path := Path;
      Receipt.Inputs (Receipt.Input_Count).Hash := Hash;
      Success := True;
   end Add_Input;

   --  Finalize and validate receipt
   procedure Finalize_Receipt (Receipt : in out Build_Receipt) is
   begin
      Receipt.Is_Valid := Receipt.Target.Length > 0;
   end Finalize_Receipt;

   --  Compute receipt core ID (simplified - would use proper hashing)
   function Compute_Receipt_Core_Id (
      Receipt : Build_Receipt) return Hash_Hex
   is
      Result : Hash_Hex := Zero_Hash;
   begin
      --  Simplified: use target hash as core ID
      --  Real implementation would hash target + epoch + inputs
      Result := Receipt.Target_Hash;
      return Result;
   end Compute_Receipt_Core_Id;

   --  Generate receipt from compilation result
   procedure Generate_Compilation_Receipt (
      Target_Path   : Path_String;
      Tool_Path     : Path_String;
      Epoch         : Epoch_Value;
      Compiled_Ok   : Boolean;
      Receipt       : out Build_Receipt)
   is
   begin
      Initialize_Receipt (Receipt, Kind_Build);
      Set_Target (Receipt, Target_Path);
      Set_Epoch (Receipt, Epoch);
      Set_Tool (Receipt, Tool_Path, Zero_Hash);
      
      if Compiled_Ok then
         Set_Status (Receipt, Receipt_Binary_Emitted);
      else
         Set_Status (Receipt, Receipt_Compilation_Failed);
      end if;
      
      Finalize_Receipt (Receipt);
   end Generate_Compilation_Receipt;

   --  Add receipt to registry
   procedure Add_To_Registry (
      Registry : in out Receipt_Registry;
      Receipt  : Build_Receipt;
      Success  : out Boolean)
   is
   begin
      if Registry.Count >= Max_Receipts then
         Success := False;
         return;
      end if;
      
      Registry.Count := Registry.Count + 1;
      Registry.Receipts (Registry.Count) := Receipt;
      Success := True;
   end Add_To_Registry;

   --  Find receipt by target path
   function Find_Receipt (
      Registry : Receipt_Registry;
      Target   : Path_String) return Natural
   is
   begin
      for I in 1 .. Registry.Count loop
         if Registry.Receipts (I).Target.Length = Target.Length then
            declare
               Match : Boolean := True;
            begin
               for J in 1 .. Target.Length loop
                  if Registry.Receipts (I).Target.Data (J) /= Target.Data (J) then
                     Match := False;
                     exit;
                  end if;
               end loop;
               if Match then
                  return I;
               end if;
            end;
         end if;
      end loop;
      return 0;
   end Find_Receipt;

end Receipt_Generator;
