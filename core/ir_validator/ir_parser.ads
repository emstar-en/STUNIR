-------------------------------------------------------------------------------
--  STUNIR IR Parser - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides IR parsing and validation.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package IR_Parser is

   --  Maximum bounds
   Max_Fields      : constant := 64;
   Max_Functions   : constant := 1_000;
   Max_Types       : constant := 500;
   Max_Name_Length : constant := 256;
   Max_Schema_Length : constant := 64;
   Max_Errors      : constant := 100;

   --  Bounded name type
   subtype Name_Length is Natural range 0 .. Max_Name_Length;
   type Bounded_Name is record
      Data   : String (1 .. Max_Name_Length) := (others => ' ');
      Length : Name_Length := 0;
   end record;

   Empty_Name : constant Bounded_Name := (Data => (others => ' '), Length => 0);

   --  Schema name type
   subtype Schema_Length is Natural range 0 .. Max_Schema_Length;
   type Schema_Name is record
      Data   : String (1 .. Max_Schema_Length) := (others => ' ');
      Length : Schema_Length := 0;
   end record;

   Empty_Schema : constant Schema_Name := (Data => (others => ' '), Length => 0);

   --  IR Schema enumeration (known schemas)
   type IR_Schema_Kind is (
      Unknown_Schema,
      IR_V1_Schema      --  stunir.ir.v1
   );

   --  Error/Warning record
   type Parse_Message is record
      Is_Error : Boolean := True;
      Message  : Bounded_Name := Empty_Name;
   end record;

   type Message_Array is array (Positive range <>) of Parse_Message;
   subtype Message_Vector is Message_Array (1 .. Max_Errors);

   --  IR Function info (simplified)
   type IR_Function_Info is record
      Name     : Bounded_Name := Empty_Name;
      Has_Body : Boolean := False;
      Param_Count : Natural := 0;
   end record;

   type Function_Array is array (Positive range <>) of IR_Function_Info;
   subtype Function_Vector is Function_Array (1 .. Max_Functions);

   --  IR Type info (simplified)
   type IR_Type_Info is record
      Name : Bounded_Name := Empty_Name;
   end record;

   type Type_Array is array (Positive range <>) of IR_Type_Info;
   subtype Type_Vector is Type_Array (1 .. Max_Types);

   --  Parse result record
   type Parse_Result is record
      Is_Valid       : Boolean := False;
      Schema         : Schema_Name := Empty_Schema;
      Schema_Kind    : IR_Schema_Kind := Unknown_Schema;
      Module_Name    : Bounded_Name := Empty_Name;
      Has_Epoch      : Boolean := False;
      Epoch          : Natural := 0;
      Has_Spec_Hash  : Boolean := False;
      Functions      : Function_Vector := (others => (others => <>));
      Function_Count : Natural := 0;
      Types          : Type_Vector := (others => (others => <>));
      Type_Count     : Natural := 0;
      Errors         : Message_Vector := (others => (others => <>));
      Error_Count    : Natural := 0;
      Warnings       : Message_Vector := (others => (others => <>));
      Warning_Count  : Natural := 0;
   end record;

   --  Create helpers
   function Make_Name (S : String) return Bounded_Name
     with
       Pre  => S'Length <= Max_Name_Length,
       Post => Make_Name'Result.Length = S'Length;

   function Make_Schema (S : String) return Schema_Name
     with
       Pre  => S'Length <= Max_Schema_Length,
       Post => Make_Schema'Result.Length = S'Length;

   --  Name comparison
   function Names_Equal (Left, Right : Bounded_Name) return Boolean;
   function Schemas_Equal (Left, Right : Schema_Name) return Boolean;

   --  Schema identification
   function Identify_Schema (S : Schema_Name) return IR_Schema_Kind;

   --  Validation functions
   function Is_Valid_Schema (S : Schema_Name) return Boolean;
   function Is_Valid_Module_Name (N : Bounded_Name) return Boolean;

   --  Required schema string for stunir.ir.v1
   V1_Schema_String : constant String := "stunir.ir.v1";

   --  Add error to result
   procedure Add_Error (
      Result  : in out Parse_Result;
      Message : Bounded_Name)
     with
       Pre => Result.Error_Count < Max_Errors;

   --  Add warning to result
   procedure Add_Warning (
      Result  : in out Parse_Result;
      Message : Bounded_Name)
     with
       Pre => Result.Warning_Count < Max_Errors;

   --  Initialize parse result
   procedure Initialize (Result : out Parse_Result);

   --  Validate IR structure (simplified - actual JSON parsing not SPARK-safe)
   procedure Validate_Structure (
      Result      : in out Parse_Result;
      Has_Schema  : Boolean;
      Has_Module  : Boolean;
      Has_Functions : Boolean;
      Strict      : Boolean := False);

   --  Add a function to the result
   procedure Add_Function (
      Result    : in out Parse_Result;
      Name      : Bounded_Name;
      Has_Body  : Boolean := False;
      Params    : Natural := 0)
     with
       Pre => Result.Function_Count < Max_Functions;

   --  Get total error count
   function Total_Errors (Result : Parse_Result) return Natural is
     (Result.Error_Count);

   --  Get total warning count
   function Total_Warnings (Result : Parse_Result) return Natural is
     (Result.Warning_Count);

end IR_Parser;
