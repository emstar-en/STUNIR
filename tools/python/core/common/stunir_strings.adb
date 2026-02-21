-------------------------------------------------------------------------------
--  STUNIR Bounded Strings - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Stunir_Strings is

   -------------------------------------------------------------------------
   --  Make_Short: Create a short string
   -------------------------------------------------------------------------
   function Make_Short (S : String) return Short_String is
      Result : Short_String;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Short;

   -------------------------------------------------------------------------
   --  To_String (Short): Convert to standard string
   -------------------------------------------------------------------------
   function To_String (S : Short_String) return String is
   begin
      return S.Data (1 .. S.Length);
   end To_String;

   -------------------------------------------------------------------------
   --  Short_Equal: Compare two short strings
   -------------------------------------------------------------------------
   function Short_Equal (Left, Right : Short_String) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Short_Equal;

   -------------------------------------------------------------------------
   --  Make_Medium: Create a medium string
   -------------------------------------------------------------------------
   function Make_Medium (S : String) return Medium_String is
      Result : Medium_String;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Medium;

   -------------------------------------------------------------------------
   --  To_String (Medium): Convert to standard string
   -------------------------------------------------------------------------
   function To_String (S : Medium_String) return String is
   begin
      return S.Data (1 .. S.Length);
   end To_String;

   -------------------------------------------------------------------------
   --  Medium_Equal: Compare two medium strings
   -------------------------------------------------------------------------
   function Medium_Equal (Left, Right : Medium_String) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Medium_Equal;

   -------------------------------------------------------------------------
   --  Make_Long: Create a long string
   -------------------------------------------------------------------------
   function Make_Long (S : String) return Long_String is
      Result : Long_String;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Long;

   -------------------------------------------------------------------------
   --  To_String (Long): Convert to standard string
   -------------------------------------------------------------------------
   function To_String (S : Long_String) return String is
   begin
      return S.Data (1 .. S.Length);
   end To_String;

   -------------------------------------------------------------------------
   --  Long_Equal: Compare two long strings
   -------------------------------------------------------------------------
   function Long_Equal (Left, Right : Long_String) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Long_Equal;

   -------------------------------------------------------------------------
   --  Make_Path: Create a path string
   -------------------------------------------------------------------------
   function Make_Path (S : String) return Path_String is
      Result : Path_String;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Path;

   -------------------------------------------------------------------------
   --  To_String (Path): Convert to standard string
   -------------------------------------------------------------------------
   function To_String (S : Path_String) return String is
   begin
      return S.Data (1 .. S.Length);
   end To_String;

   -------------------------------------------------------------------------
   --  Path_Equal: Compare two path strings
   -------------------------------------------------------------------------
   function Path_Equal (Left, Right : Path_String) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Path_Equal;

   -------------------------------------------------------------------------
   --  Concat_Short: Concatenate two short strings
   -------------------------------------------------------------------------
   function Concat_Short (Left, Right : Short_String) return Long_String is
      Result : Long_String;
   begin
      Result.Length := Left.Length + Right.Length;
      for I in 1 .. Left.Length loop
         Result.Data (I) := Left.Data (I);
      end loop;
      for I in 1 .. Right.Length loop
         Result.Data (Left.Length + I) := Right.Data (I);
      end loop;
      return Result;
   end Concat_Short;

end Stunir_Strings;
