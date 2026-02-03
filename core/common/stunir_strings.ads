-------------------------------------------------------------------------------
--  STUNIR Bounded Strings - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides bounded string types for SPARK-safe string handling.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package Stunir_Strings is

   --  Common string length bounds
   Max_Short_String  : constant := 64;
   Max_Medium_String : constant := 256;
   Max_Long_String   : constant := 1024;
   Max_Path_String   : constant := 4096;

   --  Short string (64 chars)
   subtype Short_Length is Natural range 0 .. Max_Short_String;
   type Short_String is record
      Data   : String (1 .. Max_Short_String) := (others => ' ');
      Length : Short_Length := 0;
   end record;

   Empty_Short : constant Short_String := (Data => (others => ' '), Length => 0);

   --  Medium string (256 chars)
   subtype Medium_Length is Natural range 0 .. Max_Medium_String;
   type Medium_String is record
      Data   : String (1 .. Max_Medium_String) := (others => ' ');
      Length : Medium_Length := 0;
   end record;

   Empty_Medium : constant Medium_String := (Data => (others => ' '), Length => 0);

   --  Long string (1024 chars)
   subtype Long_Length is Natural range 0 .. Max_Long_String;
   type Long_String is record
      Data   : String (1 .. Max_Long_String) := (others => ' ');
      Length : Long_Length := 0;
   end record;

   Empty_Long : constant Long_String := (Data => (others => ' '), Length => 0);

   --  Path string (4096 chars for file paths)
   subtype Path_Length is Natural range 0 .. Max_Path_String;
   type Path_String is record
      Data   : String (1 .. Max_Path_String) := (others => ' ');
      Length : Path_Length := 0;
   end record;

   Empty_Path : constant Path_String := (Data => (others => ' '), Length => 0);

   --  Short string operations
   function Make_Short (S : String) return Short_String
     with
       Pre  => S'Length <= Max_Short_String,
       Post => Make_Short'Result.Length = S'Length;

   function To_String (S : Short_String) return String
     with
       Post => To_String'Result'Length = S.Length;

   function Short_Equal (Left, Right : Short_String) return Boolean;

   --  Medium string operations
   function Make_Medium (S : String) return Medium_String
     with
       Pre  => S'Length <= Max_Medium_String,
       Post => Make_Medium'Result.Length = S'Length;

   function To_String (S : Medium_String) return String
     with
       Post => To_String'Result'Length = S.Length;

   function Medium_Equal (Left, Right : Medium_String) return Boolean;

   --  Long string operations
   function Make_Long (S : String) return Long_String
     with
       Pre  => S'Length <= Max_Long_String,
       Post => Make_Long'Result.Length = S'Length;

   function To_String (S : Long_String) return String
     with
       Post => To_String'Result'Length = S.Length;

   function Long_Equal (Left, Right : Long_String) return Boolean;

   --  Path string operations
   function Make_Path (S : String) return Path_String
     with
       Pre  => S'Length <= Max_Path_String,
       Post => Make_Path'Result.Length = S'Length;

   function To_String (S : Path_String) return String
     with
       Post => To_String'Result'Length = S.Length;

   function Path_Equal (Left, Right : Path_String) return Boolean;

   --  String concatenation (into long string)
   function Concat_Short (Left, Right : Short_String) return Long_String
     with
       Pre  => Natural (Left.Length) + Natural (Right.Length) <= Max_Long_String,
       Post => Concat_Short'Result.Length = Left.Length + Right.Length;

end Stunir_Strings;
