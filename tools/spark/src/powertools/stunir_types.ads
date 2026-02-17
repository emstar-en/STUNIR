--  stunir_types - Core types for STUNIR
--  Phase 1 Powertool for STUNIR

pragma SPARK_Mode (Off);

with Ada.Strings.Unbounded;
with Ada.Strings.Bounded;

package Stunir_Types is

   use Ada.Strings.Unbounded;

   --  Maximum JSON input size (1MB)
   Max_JSON_Length : constant := 1_048_576;

   --  Bounded string for JSON input
   package JSON_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_JSON_Length);
   subtype JSON_String is JSON_Strings.Bounded_String;

   --  Token types for JSON parsing
   type Token_Type is (
      Token_EOF,
      Token_Error,
      Token_LBrace,      -- {
      Token_RBrace,      -- }
      Token_LBracket,    -- [
      Token_RBracket,    -- ]
      Token_Colon,       -- :
      Token_Comma,       -- ,
      Token_String,
      Token_Number,
      Token_True,
      Token_False,
      Token_Null
   );

   --  Aliases for compatibility
   Token_Object_Start : constant Token_Type := Token_LBrace;
   Token_Object_End   : constant Token_Type := Token_RBrace;
   Token_Array_Start  : constant Token_Type := Token_LBracket;
   Token_Array_End    : constant Token_Type := Token_RBracket;

   --  Parser state
   type Parser_State is record
      Input       : JSON_String;
      Position    : Natural := 1;
      Line        : Natural := 1;
      Column      : Natural := 1;
      Current_Token : Token_Type := Token_EOF;
      Token_Value : Unbounded_String := Null_Unbounded_String;
   end record;

   --  Status codes
   type Status_Code is (Success, Error, EOF_Reached);

end Stunir_Types;