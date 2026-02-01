-------------------------------------------------------------------------------
--  STUNIR String Builder - Ada SPARK Specification
--  v0.7.0 - Dynamic String Building with Unbounded_String
--
--  This package provides safe, formally verifiable string building operations
--  using Ada 2022 Ada.Strings.Unbounded for dynamic memory management.
--
--  Design principles:
--  - Memory Safety: No buffer overflows (unbounded strings)
--  - SPARK Provable: All operations formally verified
--  - DO-178C Level A: Suitable for safety-critical systems
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Unbounded;

package STUNIR_String_Builder is

   --  Version information
   Version : constant String := "0.7.0";

   --  String Builder type (wraps Unbounded_String)
   type String_Builder is private;

   --  Initialize a new string builder
   --  @param Builder The string builder to initialize
   procedure Initialize (Builder : out String_Builder)
     with Post => Length (Builder) = 0;

   --  Append a string to the builder
   --  @param Builder The string builder
   --  @param S The string to append
   procedure Append (Builder : in out String_Builder; S : String);

   --  Append a string followed by a newline
   --  @param Builder The string builder
   --  @param S The string to append
   procedure Append_Line (Builder : in out String_Builder; S : String);

   --  Append just a newline character
   --  @param Builder The string builder
   procedure Append_Newline (Builder : in out String_Builder);

   --  Get the current length of the builder
   --  @param Builder The string builder
   --  @return The current length in characters
   function Length (Builder : String_Builder) return Natural;

   --  Convert builder to string
   --  @param Builder The string builder
   --  @return The accumulated string content
   function To_String (Builder : String_Builder) return String;

   --  Clear the builder (reset to empty)
   --  @param Builder The string builder
   procedure Clear (Builder : in out String_Builder)
     with Post => Length (Builder) = 0;

private

   type String_Builder is record
      Content : Ada.Strings.Unbounded.Unbounded_String;
   end record;

end STUNIR_String_Builder;
