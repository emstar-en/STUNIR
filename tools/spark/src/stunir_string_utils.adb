--  STUNIR String Utilities - Implementation
--  Centralized SPARK-compliant string operations

pragma SPARK_Mode (On);

package body STUNIR_String_Utils is

   -----------------------------------------------------------------------------
   --  Conversion Functions
   -----------------------------------------------------------------------------

   function To_String (Source : JSON_String) return String is
   begin
      return JSON_Strings.To_String (Source);
   end To_String;

   function To_JSON_String (Source : String) return JSON_String is
   begin
      return JSON_Strings.To_Bounded_String (Source);
   end To_JSON_String;

   function To_String (Source : String_Access) return String is
   begin
      if Source = null then
         return "";
      else
         return Source.all;
      end if;
   end To_String;

   function To_String_Access (Source : String) return String_Access is
   begin
      return new String'(Source);
   end To_String_Access;

   function To_String_Access (Source : JSON_String) return String_Access is
   begin
      return new String'(JSON_Strings.To_String (Source));
   end To_String_Access;

   -----------------------------------------------------------------------------
   --  SPARK-Compliant Operations
   -----------------------------------------------------------------------------

   procedure Append (Target : in out JSON_String; Source : String) is
   begin
      JSON_Strings.Append (Target, Source);
   end Append;

   function Concatenate (Left, Right : JSON_String) return JSON_String is
   begin
      return JSON_Strings."&" (Left, Right);
   end Concatenate;

   function Concatenate (Left : JSON_String; Right : String) return JSON_String is
   begin
      return JSON_Strings."&" (Left, Right);
   end Concatenate;

   -----------------------------------------------------------------------------
   --  CLI Argument Handling
   -----------------------------------------------------------------------------

   function Get_CLI_Arg (Arg : String_Access; Default : String := "") return String is
   begin
      if Arg = null then
         return Default;
      else
         return Arg.all;
      end if;
   end Get_CLI_Arg;

   function Is_Empty (Arg : String_Access) return Boolean is
   begin
      return Arg = null or else Arg.all'Length = 0;
   end Is_Empty;

   -----------------------------------------------------------------------------
   --  Verification Utilities
   -----------------------------------------------------------------------------

   function Fits_In_JSON_String (Source : String) return Boolean is
   begin
      return Source'Length <= JSON_Strings.Max_Length;
   end Fits_In_JSON_String;

   function Truncate_To_JSON_String (Source : String) return JSON_String is
      Max_Len : constant Natural := JSON_Strings.Max_Length;
   begin
      if Source'Length <= Max_Len then
         return JSON_Strings.To_Bounded_String (Source);
      else
         return JSON_Strings.To_Bounded_String (Source (Source'First .. Source'First + Max_Len - 1));
      end if;
   end Truncate_To_JSON_String;

end STUNIR_String_Utils;
