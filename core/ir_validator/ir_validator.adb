-------------------------------------------------------------------------------
--  STUNIR IR Validator - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body IR_Validator is

   -------------------------------------------------------------------------
   --  Make_Hash: Create a hash string from a string
   -------------------------------------------------------------------------
   function Make_Hash (S : String) return Hash_String is
      Result : Hash_String;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Hash;

   -------------------------------------------------------------------------
   --  Hashes_Equal: Compare two hash strings
   -------------------------------------------------------------------------
   function Hashes_Equal (Left, Right : Hash_String) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Hashes_Equal;

   -------------------------------------------------------------------------
   --  Initialize: Initialize a validation result
   -------------------------------------------------------------------------
   procedure Initialize (
      Result : out Validation_Result;
      Strict : Boolean := False)
   is
   begin
      IR_Parser.Initialize (Result.Parse_Result);
      Result.Metadata := (
         Content_Hash   => Empty_Hash,
         Function_Count => 0,
         Type_Count     => 0,
         Has_Imports    => False,
         Has_Exports    => False
      );
      Result.Is_Valid := True;  --  Assume valid until proven otherwise
      Result.Is_Strict := Strict;
   end Initialize;

   -------------------------------------------------------------------------
   --  Set_Schema: Set schema information
   -------------------------------------------------------------------------
   procedure Set_Schema (
      Result : in out Validation_Result;
      Schema : Schema_Name)
   is
   begin
      Result.Parse_Result.Schema := Schema;
      Result.Parse_Result.Schema_Kind := IR_Parser.Identify_Schema (Schema);
   end Set_Schema;

   -------------------------------------------------------------------------
   --  Set_Module: Set module name
   -------------------------------------------------------------------------
   procedure Set_Module (
      Result : in out Validation_Result;
      Name   : Bounded_Name)
   is
   begin
      Result.Parse_Result.Module_Name := Name;
   end Set_Module;

   -------------------------------------------------------------------------
   --  Set_Epoch: Set epoch value
   -------------------------------------------------------------------------
   procedure Set_Epoch (
      Result : in out Validation_Result;
      Epoch  : Natural)
   is
   begin
      Result.Parse_Result.Has_Epoch := True;
      Result.Parse_Result.Epoch := Epoch;
   end Set_Epoch;

   -------------------------------------------------------------------------
   --  Add_Validated_Function: Add a validated function
   -------------------------------------------------------------------------
   procedure Add_Validated_Function (
      Result   : in out Validation_Result;
      Name     : Bounded_Name;
      Has_Body : Boolean := False;
      Params   : Natural := 0)
   is
   begin
      IR_Parser.Add_Function (Result.Parse_Result, Name, Has_Body, Params);
      Result.Metadata.Function_Count := Result.Parse_Result.Function_Count;
   end Add_Validated_Function;

   -------------------------------------------------------------------------
   --  Set_Content_Hash: Set the content hash
   -------------------------------------------------------------------------
   procedure Set_Content_Hash (
      Result : in out Validation_Result;
      Hash   : Hash_String)
   is
   begin
      Result.Metadata.Content_Hash := Hash;
   end Set_Content_Hash;

   -------------------------------------------------------------------------
   --  Finalize_Validation: Run all validation checks
   -------------------------------------------------------------------------
   procedure Finalize_Validation (Result : in out Validation_Result) is
      Has_Schema    : Boolean;
      Has_Module    : Boolean;
      Has_Functions : Boolean;
   begin
      --  Check what fields are present
      Has_Schema := Result.Parse_Result.Schema.Length > 0;
      Has_Module := Result.Parse_Result.Module_Name.Length > 0;
      Has_Functions := Result.Parse_Result.Function_Count > 0;

      --  Run structure validation
      IR_Parser.Validate_Structure (
         Result.Parse_Result,
         Has_Schema    => Has_Schema,
         Has_Module    => Has_Module,
         Has_Functions => Has_Functions,
         Strict        => Result.Is_Strict
      );

      --  Update overall validity
      Result.Is_Valid := Result.Parse_Result.Is_Valid;
   end Finalize_Validation;

end IR_Validator;
