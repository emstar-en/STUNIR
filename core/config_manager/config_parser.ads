-------------------------------------------------------------------------------
--  STUNIR Config Parser - Ada SPARK Specification
--  Part of Phase 2 SPARK Migration
--
--  This package provides configuration parsing from environment and files.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Stunir_Strings; use Stunir_Strings;
with Build_Config;   use Build_Config;

package Config_Parser is

   --  Maximum command line arguments
   Max_Args : constant := 64;

   --  Argument types
   type Arg_Kind is (
      Arg_Flag,      --  --flag
      Arg_Value,     --  --name=value or --name value
      Arg_Positional --  plain value
   );

   --  Parsed argument
   type Parsed_Arg is record
      Kind   : Arg_Kind := Arg_Positional;
      Name   : Short_String := Empty_Short;
      Value  : Medium_String := Empty_Medium;
   end record;

   type Arg_Array is array (Positive range <>) of Parsed_Arg;
   subtype Arg_Vector is Arg_Array (1 .. Max_Args);

   --  Parse result
   type Parse_Result is record
      Args      : Arg_Vector := (others => (others => <>));
      Arg_Count : Natural := 0;
      Has_Error : Boolean := False;
      Error_Msg : Medium_String := Empty_Medium;
   end record;

   --  Parse environment variable for profile
   procedure Parse_Profile_Env (
      Profile : out Build_Profile;
      Found   : out Boolean);

   --  Parse configuration from environment
   procedure Parse_From_Environment (
      Config  : out Configuration;
      Success : out Boolean);

   --  Apply command line argument to config
   procedure Apply_Argument (
      Config : in out Configuration;
      Arg    : Parsed_Arg)
     with
       Pre => Arg.Name.Length > 0 or Arg.Kind = Arg_Positional;

   --  Apply all parsed arguments to config
   procedure Apply_All_Arguments (
      Config  : in out Configuration;
      Result  : Parse_Result);

   --  Check for flag in arguments
   function Has_Flag (
      Result : Parse_Result;
      Name   : String) return Boolean;

   --  Get value for argument
   function Get_Value (
      Result  : Parse_Result;
      Name    : String;
      Default : String := "") return Medium_String;

end Config_Parser;
