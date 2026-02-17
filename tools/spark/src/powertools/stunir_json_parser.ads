--  stunir_json_parser - JSON parsing utilities
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Strings.Unbounded;
with Stunir_Types;

package Stunir_JSON_Parser is

   use Ada.Strings.Unbounded;
   use Stunir_Types;

   --  Token constants for compatibility
   Token_EOF : constant Token_Type := Stunir_Types.Token_EOF;

   --  Initialize parser with input string
   procedure Initialize_Parser
     (State  : in out Parser_State;
      Input  : JSON_String;
      Status : out Status_Code);

   --  Get next token from input
   procedure Next_Token
     (State  : in out Parser_State;
      Status : out Status_Code);

   --  Simple validation function
   function Validate_JSON (Input : String) return Boolean;

   --  Skip a JSON value (used for parsing)
   procedure Skip_Value
     (State  : in out Parser_State;
      Status : out Status_Code);

   --  Expect a specific token
   procedure Expect_Token
     (State    : in out Parser_State;
      Expected : Token_Type;
      Status   : out Status_Code);

end Stunir_JSON_Parser;