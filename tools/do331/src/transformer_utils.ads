--  STUNIR DO-331 Transformer Utilities Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides utility functions for the IR-to-Model transformer.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;

package Transformer_Utils is

   --  ============================================================
   --  Name Transformation
   --  ============================================================

   --  Normalize a name to valid SysML identifier format
   function Normalize_Name (Name : String) return String
     with Pre  => Name'Length > 0 and Name'Length <= Max_Name_Length,
          Post => Normalize_Name'Result'Length > 0 and
                  Normalize_Name'Result'Length <= Max_Name_Length;

   --  Convert IR identifier to SysML-compatible name
   --  (CamelCase, valid characters)
   function To_SysML_Identifier (IR_Name : String) return String
     with Pre => IR_Name'Length > 0 and IR_Name'Length <= Max_Name_Length;

   --  Check if a name is a valid SysML identifier
   function Is_Valid_SysML_Identifier (Name : String) return Boolean
     with Pre => Name'Length > 0;

   --  Convert snake_case to CamelCase
   function Snake_To_Camel (Name : String) return String
     with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length;

   --  ============================================================
   --  String Utilities
   --  ============================================================

   --  Escape special characters in a string for SysML output
   function Escape_String (S : String) return String
     with Pre => S'Length >= 0;

   --  Check if character is alphanumeric or underscore
   function Is_Identifier_Char (C : Character) return Boolean;

   --  Check if character is a letter
   function Is_Letter (C : Character) return Boolean;

   --  Check if character is a digit
   function Is_Digit (C : Character) return Boolean;

   --  Convert character to uppercase
   function To_Upper (C : Character) return Character;

   --  Convert character to lowercase
   function To_Lower (C : Character) return Character;

   --  ============================================================
   --  Hash Utilities
   --  ============================================================

   --  Simple hash function for strings (for ID generation)
   function Simple_Hash (S : String) return Natural
     with Pre => S'Length > 0;

   --  Format a hex digit
   function Hex_Digit (N : Natural) return Character
     with Pre => N < 16;

   --  ============================================================
   --  Path Utilities
   --  ============================================================

   --  Build a qualified path from parent and child names
   function Build_Qualified_Path (
      Parent_Path : String;
      Child_Name  : String
   ) return String
     with Pre => Child_Name'Length > 0;

   --  Extract the last segment from a path
   function Get_Last_Segment (Path : String) return String
     with Pre => Path'Length > 0;

   --  ============================================================
   --  Type Utilities
   --  ============================================================

   --  Map common IR type names to canonical form
   function Canonicalize_Type_Name (Type_Name : String) return String
     with Pre => Type_Name'Length > 0;

   --  Check if type is a primitive type
   function Is_Primitive_Type (Type_Name : String) return Boolean
     with Pre => Type_Name'Length > 0;

   --  ============================================================
   --  Numeric Utilities
   --  ============================================================

   --  Convert natural to string
   function Natural_To_String (N : Natural) return String;

   --  Get current epoch (stub - returns fixed value for determinism)
   function Get_Current_Epoch return Natural;

end Transformer_Utils;
