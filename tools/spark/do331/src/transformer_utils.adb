--  STUNIR DO-331 Transformer Utilities Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Transformer_Utils is

   --  ============================================================
   --  Character Utilities
   --  ============================================================

   function Is_Letter (C : Character) return Boolean is
   begin
      return C in 'A' .. 'Z' or C in 'a' .. 'z';
   end Is_Letter;

   function Is_Digit (C : Character) return Boolean is
   begin
      return C in '0' .. '9';
   end Is_Digit;

   function Is_Identifier_Char (C : Character) return Boolean is
   begin
      return Is_Letter (C) or Is_Digit (C) or C = '_';
   end Is_Identifier_Char;

   function To_Upper (C : Character) return Character is
   begin
      if C in 'a' .. 'z' then
         return Character'Val (Character'Pos (C) - 32);
      else
         return C;
      end if;
   end To_Upper;

   function To_Lower (C : Character) return Character is
   begin
      if C in 'A' .. 'Z' then
         return Character'Val (Character'Pos (C) + 32);
      else
         return C;
      end if;
   end To_Lower;

   --  ============================================================
   --  Name Transformation
   --  ============================================================

   function Normalize_Name (Name : String) return String is
      Result : String (1 .. Name'Length);
      J      : Natural := 0;
   begin
      for I in Name'Range loop
         if Is_Identifier_Char (Name (I)) then
            J := J + 1;
            Result (J) := Name (I);
         elsif Name (I) = '-' or Name (I) = ' ' then
            J := J + 1;
            Result (J) := '_';
         end if;
      end loop;

      if J = 0 then
         return "unnamed";
      else
         return Result (1 .. J);
      end if;
   end Normalize_Name;

   function To_SysML_Identifier (IR_Name : String) return String is
      Normalized : constant String := Normalize_Name (IR_Name);
   begin
      return Snake_To_Camel (Normalized);
   end To_SysML_Identifier;

   function Is_Valid_SysML_Identifier (Name : String) return Boolean is
   begin
      if Name'Length = 0 then
         return False;
      end if;

      --  First character must be letter or underscore
      if not (Is_Letter (Name (Name'First)) or Name (Name'First) = '_') then
         return False;
      end if;

      --  Remaining characters must be alphanumeric or underscore
      for I in Name'First + 1 .. Name'Last loop
         if not Is_Identifier_Char (Name (I)) then
            return False;
         end if;
      end loop;

      return True;
   end Is_Valid_SysML_Identifier;

   function Snake_To_Camel (Name : String) return String is
      Result       : String (1 .. Name'Length);
      J            : Natural := 0;
      Capitalize   : Boolean := True;
   begin
      for I in Name'Range loop
         if Name (I) = '_' then
            Capitalize := True;
         else
            J := J + 1;
            if Capitalize then
               Result (J) := To_Upper (Name (I));
               Capitalize := False;
            else
               Result (J) := Name (I);
            end if;
         end if;
      end loop;

      if J = 0 then
         return "Unnamed";
      else
         return Result (1 .. J);
      end if;
   end Snake_To_Camel;

   --  ============================================================
   --  String Utilities
   --  ============================================================

   function Escape_String (S : String) return String is
      Result : String (1 .. S'Length * 2);  -- Worst case: all escaped
      J      : Natural := 0;
   begin
      for I in S'Range loop
         case S (I) is
            when '"' =>
               J := J + 1;
               Result (J) := '\\';
               J := J + 1;
               Result (J) := '"';
            when '\\' =>
               J := J + 1;
               Result (J) := '\\';
               J := J + 1;
               Result (J) := '\\';
            when ASCII.LF =>
               J := J + 1;
               Result (J) := '\\';
               J := J + 1;
               Result (J) := 'n';
            when ASCII.CR =>
               J := J + 1;
               Result (J) := '\\';
               J := J + 1;
               Result (J) := 'r';
            when ASCII.HT =>
               J := J + 1;
               Result (J) := '\\';
               J := J + 1;
               Result (J) := 't';
            when others =>
               J := J + 1;
               Result (J) := S (I);
         end case;
      end loop;

      return Result (1 .. J);
   end Escape_String;

   --  ============================================================
   --  Hash Utilities
   --  ============================================================

   function Simple_Hash (S : String) return Natural is
      H : Natural := 5381;
   begin
      for I in S'Range loop
         --  Simple djb2 hash variant
         if H < Natural'Last / 33 - 256 then
            H := H * 33 + Character'Pos (S (I));
         else
            --  Prevent overflow by wrapping
            H := H mod 1_000_000 + Character'Pos (S (I));
         end if;
      end loop;
      return H;
   end Simple_Hash;

   function Hex_Digit (N : Natural) return Character is
      Hex_Chars : constant String := "0123456789abcdef";
   begin
      return Hex_Chars (N + 1);
   end Hex_Digit;

   --  ============================================================
   --  Path Utilities
   --  ============================================================

   function Build_Qualified_Path (
      Parent_Path : String;
      Child_Name  : String
   ) return String is
   begin
      if Parent_Path'Length = 0 then
         return Child_Name;
      else
         return Parent_Path & "::" & Child_Name;
      end if;
   end Build_Qualified_Path;

   function Get_Last_Segment (Path : String) return String is
      Last_Sep : Natural := 0;
   begin
      --  Find last :: separator
      for I in Path'First .. Path'Last - 1 loop
         if Path (I) = ':' and then I < Path'Last and then Path (I + 1) = ':' then
            Last_Sep := I + 2;
         end if;
      end loop;

      if Last_Sep = 0 then
         return Path;
      elsif Last_Sep <= Path'Last then
         return Path (Last_Sep .. Path'Last);
      else
         return Path;
      end if;
   end Get_Last_Segment;

   --  ============================================================
   --  Type Utilities
   --  ============================================================

   function Canonicalize_Type_Name (Type_Name : String) return String is
   begin
      if Type_Name = "int" or Type_Name = "i32" or Type_Name = "int32" then
         return "Integer";
      elsif Type_Name = "i64" or Type_Name = "int64" then
         return "Integer";
      elsif Type_Name = "float" or Type_Name = "f32" or Type_Name = "float32" then
         return "Real";
      elsif Type_Name = "double" or Type_Name = "f64" or Type_Name = "float64" then
         return "Real";
      elsif Type_Name = "bool" or Type_Name = "boolean" then
         return "Boolean";
      elsif Type_Name = "str" or Type_Name = "string" then
         return "String";
      elsif Type_Name = "void" or Type_Name = "unit" or Type_Name = "()" then
         return "Anything";
      else
         return Type_Name;
      end if;
   end Canonicalize_Type_Name;

   function Is_Primitive_Type (Type_Name : String) return Boolean is
      Canon : constant String := Canonicalize_Type_Name (Type_Name);
   begin
      return Canon = "Integer" or Canon = "Real" or
             Canon = "Boolean" or Canon = "String" or
             Canon = "Natural" or Canon = "Positive" or
             Canon = "Anything";
   end Is_Primitive_Type;

   --  ============================================================
   --  Numeric Utilities
   --  ============================================================

   function Natural_To_String (N : Natural) return String is
      Temp   : Natural := N;
      Result : String (1 .. 12);  -- Enough for any Natural
      J      : Natural := Result'Last + 1;
   begin
      if N = 0 then
         return "0";
      end if;

      while Temp > 0 and J > 1 loop
         J := J - 1;
         Result (J) := Character'Val (Character'Pos ('0') + Temp mod 10);
         Temp := Temp / 10;
      end loop;

      return Result (J .. Result'Last);
   end Natural_To_String;

   function Get_Current_Epoch return Natural is
   begin
      --  For determinism, we return a fixed value
      --  The actual epoch should be passed from environment
      return 1738108800;  -- 2026-01-29T00:00:00Z
   end Get_Current_Epoch;

end Transformer_Utils;
