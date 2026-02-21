--  STUNIR DO-331 SysML 2.0 Formatter Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body SysML_Formatter is

   --  ============================================================
   --  Indentation
   --  ============================================================

   function Indent (Level : Natural; Size : Positive := Default_Indent_Size) return String is
      Total_Spaces : constant Natural := Level * Size;
   begin
      if Total_Spaces = 0 then
         return "";
      else
         declare
            Result : String (1 .. Total_Spaces) := (others => ' ');
         begin
            return Result;
         end;
      end if;
   end Indent;

   --  ============================================================
   --  Comment Formatting
   --  ============================================================

   function Format_Doc_Comment (
      Text   : String;
      Indent : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & "doc /* " & Text & " */";
   end Format_Doc_Comment;

   function Format_Line_Comment (
      Text   : String;
      Indent : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & "// " & Text;
   end Format_Line_Comment;

   function Format_Block_Comment (
      Text   : String;
      Indent : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & "/* " & Text & " */";
   end Format_Block_Comment;

   --  ============================================================
   --  Block Formatting
   --  ============================================================

   function Format_Block_Start (
      Keyword : String;
      Name    : String;
      Indent  : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & Keyword & " " & Name & " {";
   end Format_Block_Start;

   function Format_Block_End (Indent : Natural) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & "}";
   end Format_Block_End;

   --  ============================================================
   --  Statement Formatting
   --  ============================================================

   function Format_Statement (
      Statement : String;
      Indent    : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & Statement & ";";
   end Format_Statement;

   function Format_Import (
      Import_Path : String;
      Indent      : Natural
   ) return String is
      Ind : constant String := SysML_Formatter.Indent (Indent);
   begin
      return Ind & "import " & Import_Path & ";";
   end Format_Import;

   --  ============================================================
   --  Expression Formatting
   --  ============================================================

   function Format_Expression (Expr : String) return String is
      Needs_Parens : Boolean := False;
   begin
      --  Check if expression needs parentheses
      --  (contains spaces and no outer parens)
      for I in Expr'Range loop
         if Expr (I) = ' ' then
            Needs_Parens := True;
            exit;
         end if;
      end loop;
      
      if Needs_Parens and then Expr (Expr'First) /= '(' then
         return "(" & Expr & ")";
      else
         return Expr;
      end if;
   end Format_Expression;

   function Format_Qualified_Name (
      Parts : String
   ) return String is
      Result : String (1 .. Parts'Length * 2);  -- Worst case with ::
      J      : Natural := 0;
   begin
      for I in Parts'Range loop
         if Parts (I) = '.' or Parts (I) = '/' then
            --  Convert to SysML separator
            if J + 2 <= Result'Length then
               J := J + 1;
               Result (J) := ':';
               J := J + 1;
               Result (J) := ':';
            end if;
         else
            J := J + 1;
            if J <= Result'Length then
               Result (J) := Parts (I);
            end if;
         end if;
      end loop;
      
      if J = 0 then
         return Parts;
      else
         return Result (1 .. J);
      end if;
   end Format_Qualified_Name;

   --  ============================================================
   --  String Utilities
   --  ============================================================

   function Escape_String_Literal (S : String) return String is
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
   end Escape_String_Literal;

   function Quote_String (S : String) return String is
   begin
      return "\"" & Escape_String_Literal (S) & "\"";
   end Quote_String;

   --  ============================================================
   --  Line Wrapping
   --  ============================================================

   function Needs_Wrapping (
      Line      : String;
      Max_Width : Positive
   ) return Boolean is
   begin
      return Line'Length > Max_Width;
   end Needs_Wrapping;

   function Wrap_Line (
      Line       : String;
      Max_Width  : Positive;
      Indent_Str : String
   ) return String is
      pragma Unreferenced (Indent_Str);
      Break_Point : Natural := 0;
   begin
      if Line'Length <= Max_Width then
         return Line;
      end if;
      
      --  Find a good break point (at a space)
      for I in reverse Line'First .. Line'First + Max_Width - 1 loop
         if Line (I) = ' ' then
            Break_Point := I;
            exit;
         end if;
      end loop;
      
      if Break_Point = 0 then
         --  No space found, just cut at max width
         return Line (Line'First .. Line'First + Max_Width - 1);
      else
         return Line (Line'First .. Break_Point - 1);
      end if;
   end Wrap_Line;

end SysML_Formatter;
