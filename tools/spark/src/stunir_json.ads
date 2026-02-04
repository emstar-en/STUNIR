-------------------------------------------------------------------------------
--  STUNIR JSON - Ada SPARK Specification
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Lightweight JSON file IO utilities for SPARK tools.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_JSON is

   Max_JSON_Size : constant := 1_048_576;
   Max_Hash_Length : constant := 64;

   package JSON_Buffers is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_JSON_Size);
   subtype JSON_String is JSON_Buffers.Bounded_String;

   package Hash_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length
     (Max => Max_Hash_Length);
   subtype Hash_String is Hash_Strings.Bounded_String;

   procedure Read_JSON_File
     (Path    : String;
      Content : out JSON_String;
      Success : out Boolean)
   with
     Pre => Path'Length > 0 and Path'Length <= 512,
     Post => (if Success then JSON_Buffers.Length (Content) > 0);

   procedure Write_JSON_File
     (Path    : String;
      Content : String;
      Success : out Boolean)
   with
     Pre => Path'Length > 0 and Content'Length <= Max_JSON_Size;

   procedure Compute_File_Hash
     (Path    : String;
      Hash    : out Hash_String;
      Success : out Boolean)
   with
     Pre => Path'Length > 0,
     Post => (if Success then Hash_Strings.Length (Hash) = Max_Hash_Length);

end STUNIR_JSON;
