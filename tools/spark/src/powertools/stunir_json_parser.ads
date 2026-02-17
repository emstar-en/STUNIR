--  STUNIR_JSON_Parser - JSON parsing utilities for powertools
--  SPARK-compliant parser for CLI tooling
--
--  ARCHITECTURE: Full SPARK compliance throughout Ada codebase.
--  Lightweight compared to main SPARK parser, but still formally verifiable.

pragma SPARK_Mode (On);

with STUNIR_Types;

package STUNIR_JSON_Parser is

   use STUNIR_Types;

   pragma Pure;

   --  Token constants for compatibility
   Token_EOF : constant Token_Type := STUNIR_Types.Token_EOF;

   --  Initialize parser with input string
   procedure Initialize_Parser
     (State  : out Parser_State;
      Input  : in JSON_String;
      Status : out Status_Code)
   with
      Pre  => JSON_Strings.Length (Input) > 0,
      Post => (if Status = Success then State.Position = 1);

   --  Get next token from input
   procedure Next_Token
     (State  : in out Parser_State;
      Status : out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length;

   --  Simple validation function
   function Validate_JSON (Input : String) return Boolean;

   --  Skip a JSON value (used for parsing)
   procedure Skip_Value
     (State  : in out Parser_State;
      Status : out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length;

   --  Expect a specific token
   procedure Expect_Token
     (State    : in out Parser_State;
      Expected : in Token_Type;
      Status   : out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => (if Status = Success then State.Current_Token = Expected);

end STUNIR_JSON_Parser;
