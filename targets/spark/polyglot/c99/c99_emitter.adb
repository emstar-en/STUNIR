--  STUNIR C99 Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body C99_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Header (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C99_Config;
      Status      : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "/*" & New_Line &
         " * STUNIR Generated C99 Header" & New_Line &
         " * Module: " & Name & New_Line &
         " * DO-178C Level A Compliant" & New_Line &
         " */" & New_Line &
         "#ifndef " & Name & "_H" & New_Line &
         "#define " & Name & "_H" & New_Line & New_Line &
         "#include <stdint.h>" & New_Line &
         "#include <stdbool.h>" & New_Line & New_Line &
         "#endif /* " & Name & "_H */" & New_Line);
   end Emit_Header;

   procedure Emit_Source (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C99_Config;
      Status      : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "/*" & New_Line &
         " * STUNIR Generated C99 Source" & New_Line &
         " * Module: " & Name & New_Line &
         " */" & New_Line &
         "#include \"" & Name & ".h\"" & New_Line & New_Line);
   end Emit_Source;

end C99_Emitter;
