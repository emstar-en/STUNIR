--  STUNIR String Utilities - Centralized SPARK-compliant string operations
--  Provides conversion, handling, and verification for all string types
--  Eliminates repetitive per-tool string conversion code

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;
with GNAT.Strings;
with STUNIR_Types;

package STUNIR_String_Utils is

   --  Re-export JSON_String for convenience
   subtype JSON_String is STUNIR_Types.JSON_String;
   package JSON_Strings renames STUNIR_Types.JSON_Strings;

   --  String_Access type for CLI arguments (GNAT.Command_Line compatibility)
   subtype String_Access is GNAT.Strings.String_Access;

   -----------------------------------------------------------------------------
   --  Conversion Functions
   -----------------------------------------------------------------------------

   --  Convert bounded JSON_String to String
   function To_String (Source : JSON_String) return String
     with Post => To_String'Result'Length <= JSON_Strings.Max_Length;

   --  Convert String to bounded JSON_String (with length check)
   function To_JSON_String (Source : String) return JSON_String
     with Pre => Source'Length <= JSON_Strings.Max_Length;

   --  Convert String_Access to String (safely handle null)
   function To_String (Source : String_Access) return String
     with Post => (if Source = null then To_String'Result'Length = 0);

   --  Convert String to String_Access (allocates new string)
   function To_String_Access (Source : String) return String_Access
     with Post => (To_String_Access'Result /= null and then
                   To_String_Access'Result.all = Source);

   --  Convert JSON_String to String_Access
   function To_String_Access (Source : JSON_String) return String_Access;

   -----------------------------------------------------------------------------
   --  SPARK-Compliant Operations
   -----------------------------------------------------------------------------

   --  Append string to JSON_String (qualified to avoid ambiguity)
   procedure Append (Target : in out JSON_String; Source : String)
     with Pre => (JSON_Strings.Length (Target) + Source'Length <= 
                  JSON_Strings.Max_Length);

   --  Concatenate two JSON_Strings
   function Concatenate (Left, Right : JSON_String) return JSON_String
     with Pre => (JSON_Strings.Length (Left) + JSON_Strings.Length (Right) <= 
                  JSON_Strings.Max_Length);

   --  Concatenate JSON_String with String
   function Concatenate (Left : JSON_String; Right : String) return JSON_String
     with Pre => (JSON_Strings.Length (Left) + Right'Length <= 
                  JSON_Strings.Max_Length);

   -----------------------------------------------------------------------------
   --  CLI Argument Handling
   -----------------------------------------------------------------------------

   --  Safely dereference String_Access for CLI arguments
   function Get_CLI_Arg (Arg : String_Access; Default : String := "") return String
     with Post => (if Arg = null then Get_CLI_Arg'Result = Default
                   else Get_CLI_Arg'Result = Arg.all);

   --  Check if String_Access is null or empty
   function Is_Empty (Arg : String_Access) return Boolean
     with Post => Is_Empty'Result = (Arg = null or else Arg.all'Length = 0);

   -----------------------------------------------------------------------------
   --  Verification Utilities
   -----------------------------------------------------------------------------

   --  Check if string fits in JSON_String bounds
   function Fits_In_JSON_String (Source : String) return Boolean
     with Post => Fits_In_JSON_String'Result = 
                  (Source'Length <= JSON_Strings.Max_Length);

   --  Safely truncate string to fit JSON_String if needed
   function Truncate_To_JSON_String (Source : String) return JSON_String;

end STUNIR_String_Utils;
