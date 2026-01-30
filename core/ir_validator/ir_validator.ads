-------------------------------------------------------------------------------
--  STUNIR IR Validator - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides deep IR validation against schema.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with IR_Parser; use IR_Parser;

package IR_Validator is

   --  Maximum hash length (SHA-256 hex = 64 chars)
   Max_Hash_Length : constant := 64;

   --  Hash string type
   subtype Hash_Length is Natural range 0 .. Max_Hash_Length;
   type Hash_String is record
      Data   : String (1 .. Max_Hash_Length) := (others => '0');
      Length : Hash_Length := 0;
   end record;

   Empty_Hash : constant Hash_String := (Data => (others => '0'), Length => 0);

   --  Validation metadata record
   type Validation_Metadata is record
      Content_Hash  : Hash_String := Empty_Hash;
      Function_Count: Natural := 0;
      Type_Count    : Natural := 0;
      Has_Imports   : Boolean := False;
      Has_Exports   : Boolean := False;
   end record;

   --  Validation result record
   type Validation_Result is record
      Parse_Result  : IR_Parser.Parse_Result;
      Metadata      : Validation_Metadata;
      Is_Valid      : Boolean := False;
      Is_Strict     : Boolean := False;
   end record;

   --  Initialize validation result
   procedure Initialize (
      Result : out Validation_Result;
      Strict : Boolean := False);

   --  Set schema information
   procedure Set_Schema (
      Result : in Out Validation_Result;
      Schema : Schema_Name);

   --  Set module name
   procedure Set_Module (
      Result : in Out Validation_Result;
      Name   : Bounded_Name);

   --  Set epoch
   procedure Set_Epoch (
      Result : in Out Validation_Result;
      Epoch  : Natural);

   --  Add a validated function
   procedure Add_Validated_Function (
      Result   : in Out Validation_Result;
      Name     : Bounded_Name;
      Has_Body : Boolean := False;
      Params   : Natural := 0)
     with
       Pre => Result.Parse_Result.Function_Count < Max_Functions;

   --  Compute content hash (simulated - actual hash needs external call)
   procedure Set_Content_Hash (
      Result : in Out Validation_Result;
      Hash   : Hash_String);

   --  Finalize validation (run all checks)
   procedure Finalize_Validation (Result : in Out Validation_Result);

   --  Query functions
   function Is_Valid (Result : Validation_Result) return Boolean is
     (Result.Is_Valid);

   function Get_Schema (Result : Validation_Result) return Schema_Name is
     (Result.Parse_Result.Schema);

   function Get_Module (Result : Validation_Result) return Bounded_Name is
     (Result.Parse_Result.Module_Name);

   function Get_Function_Count (Result : Validation_Result) return Natural is
     (Result.Parse_Result.Function_Count);

   function Get_Error_Count (Result : Validation_Result) return Natural is
     (Result.Parse_Result.Error_Count);

   function Get_Warning_Count (Result : Validation_Result) return Natural is
     (Result.Parse_Result.Warning_Count);

   function Get_Content_Hash (Result : Validation_Result) return Hash_String is
     (Result.Metadata.Content_Hash);

   --  Utility: Create hash string
   function Make_Hash (S : String) return Hash_String
     with
       Pre  => S'Length <= Max_Hash_Length,
       Post => Make_Hash'Result.Length = S'Length;

   --  Utility: Compare hash strings
   function Hashes_Equal (Left, Right : Hash_String) return Boolean;

end IR_Validator;
