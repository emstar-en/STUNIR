--  STUNIR_Types - Core types for STUNIR powertools
--  SPARK-compliant types for CLI tooling
--  
--  ARCHITECTURE: Full SPARK compliance throughout Ada codebase.
--  For rapid prototyping, use Python or Rust pipelines instead.

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Types is

   --  Note: Cannot use Pure pragma due to dependency on Ada.Strings.Bounded

   --  Maximum JSON input size (1MB) - matches main SPARK
   Max_JSON_Length : constant := 1_048_576;

   --  Bounded string for JSON input - SPARK-compliant
   package JSON_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_JSON_Length);
   subtype JSON_String is JSON_Strings.Bounded_String;

   --  Token types for JSON parsing - orthographic with main SPARK
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

   --  Aliases for readability
   Token_Object_Start : constant Token_Type := Token_LBrace;
   Token_Object_End   : constant Token_Type := Token_RBrace;
   Token_Array_Start  : constant Token_Type := Token_LBracket;
   Token_Array_End    : constant Token_Type := Token_RBracket;

   --  Parser state - SPARK-compliant with bounded strings only
   type Parser_State is record
      Input         : JSON_String;
      Position      : Natural := 1;
      Line          : Natural := 1;
      Column        : Natural := 1;
      Current_Token : Token_Type := Token_EOF;
      Token_Value   : JSON_String;  -- Bounded, not Unbounded
   end record;

   --  Status codes - simplified but SPARK-compliant
   type Status_Code is (Success, Error, EOF_Reached);

end STUNIR_Types;
