-------------------------------------------------------------------------------
--  STUNIR String Builder - Ada SPARK Implementation
--  v0.7.0 - Dynamic String Building with Unbounded_String
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body STUNIR_String_Builder is

   --  Newline character constant
   NL : constant String := [1 => Character'Val (10)];

   --  Initialize a new string builder
   procedure Initialize (Builder : out String_Builder) is
   begin
      Builder.Content := Null_Unbounded_String;
   end Initialize;

   --  Append a string to the builder
   procedure Append (Builder : in out String_Builder; S : String) is
   begin
      Ada.Strings.Unbounded.Append (Builder.Content, S);
   end Append;

   --  Append a string followed by a newline
   procedure Append_Line (Builder : in out String_Builder; S : String) is
   begin
      Ada.Strings.Unbounded.Append (Builder.Content, S & NL);
   end Append_Line;

   --  Append just a newline character
   procedure Append_Newline (Builder : in out String_Builder) is
   begin
      Ada.Strings.Unbounded.Append (Builder.Content, NL);
   end Append_Newline;

   --  Get the current length of the builder
   function Length (Builder : String_Builder) return Natural is
   begin
      return Ada.Strings.Unbounded.Length (Builder.Content);
   end Length;

   --  Convert builder to string
   function To_String (Builder : String_Builder) return String is
   begin
      return Ada.Strings.Unbounded.To_String (Builder.Content);
   end To_String;

   --  Clear the builder (reset to empty)
   procedure Clear (Builder : in out String_Builder) is
   begin
      Builder.Content := Null_Unbounded_String;
   end Clear;

end STUNIR_String_Builder;
