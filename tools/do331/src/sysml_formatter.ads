--  STUNIR DO-331 SysML 2.0 Formatter Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides formatting utilities for SysML 2.0 output.

pragma SPARK_Mode (On);

package SysML_Formatter is

   --  ============================================================
   --  Constants
   --  ============================================================

   Default_Indent_Size : constant := 4;
   Max_Indent_Level    : constant := 20;
   Max_Line_Length     : constant := 120;

   --  ============================================================
   --  Indentation
   --  ============================================================

   --  Create indentation string of specified level
   function Indent (Level : Natural; Size : Positive := Default_Indent_Size) return String
     with Pre => Level <= Max_Indent_Level;

   --  ============================================================
   --  Comment Formatting
   --  ============================================================

   --  Format a documentation comment
   function Format_Doc_Comment (
      Text   : String;
      Indent : Natural
   ) return String
     with Pre => Text'Length > 0 and Indent <= Max_Indent_Level;

   --  Format a line comment
   function Format_Line_Comment (
      Text   : String;
      Indent : Natural
   ) return String
     with Pre => Text'Length > 0 and Indent <= Max_Indent_Level;

   --  Format a multi-line comment
   function Format_Block_Comment (
      Text   : String;
      Indent : Natural
   ) return String
     with Pre => Text'Length > 0 and Indent <= Max_Indent_Level;

   --  ============================================================
   --  Block Formatting
   --  ============================================================

   --  Format block opening (e.g., "package Name {")
   function Format_Block_Start (
      Keyword : String;
      Name    : String;
      Indent  : Natural
   ) return String
     with Pre => Keyword'Length > 0 and Name'Length > 0 and Indent <= Max_Indent_Level;

   --  Format block closing (just "}")
   function Format_Block_End (Indent : Natural) return String
     with Pre => Indent <= Max_Indent_Level;

   --  ============================================================
   --  Statement Formatting
   --  ============================================================

   --  Format a simple statement with semicolon
   function Format_Statement (
      Statement : String;
      Indent    : Natural
   ) return String
     with Pre => Statement'Length > 0 and Indent <= Max_Indent_Level;

   --  Format an import statement
   function Format_Import (
      Import_Path : String;
      Indent      : Natural
   ) return String
     with Pre => Import_Path'Length > 0 and Indent <= Max_Indent_Level;

   --  ============================================================
   --  Expression Formatting
   --  ============================================================

   --  Format an expression (parenthesize if needed)
   function Format_Expression (Expr : String) return String
     with Pre => Expr'Length > 0;

   --  Format a qualified name (path::to::element)
   function Format_Qualified_Name (
      Parts : String  --  Dot or slash separated
   ) return String
     with Pre => Parts'Length > 0;

   --  ============================================================
   --  String Utilities
   --  ============================================================

   --  Escape special characters for string literals
   function Escape_String_Literal (S : String) return String;

   --  Quote a string
   function Quote_String (S : String) return String;

   --  ============================================================
   --  Line Wrapping
   --  ============================================================

   --  Check if line needs wrapping
   function Needs_Wrapping (
      Line      : String;
      Max_Width : Positive
   ) return Boolean
     with Pre => Line'Length > 0;

   --  Wrap a long line (returns first part, up to max width)
   function Wrap_Line (
      Line       : String;
      Max_Width  : Positive;
      Indent_Str : String
   ) return String
     with Pre => Line'Length > 0;

end SysML_Formatter;
