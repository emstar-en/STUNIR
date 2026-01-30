--  STUNIR Scheme Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Scheme_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Library (
      Library_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Scheme_Config;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Library_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         ";; STUNIR Generated Scheme Library" & New_Line &
         ";; Library: " & Name & New_Line &
         ";; DO-178C Level A Compliant" & New_Line & New_Line);

      case Config.Standard is
         when R5RS =>
            null;  --  R5RS has no library system
         when R6RS =>
            Content_Strings.Append (Content,
               "(library (" & Name & ")" & New_Line &
               "  (export)" & New_Line &
               "  (import (rnrs)))" & New_Line & New_Line);
         when R7RS =>
            Content_Strings.Append (Content,
               "(define-library (" & Name & ")" & New_Line &
               "  (export)" & New_Line &
               "  (import (scheme base)))" & New_Line & New_Line);
      end case;
   end Emit_Library;

   procedure Emit_Define (
      Name      : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Func_Name : constant String := Identifier_Strings.To_String (Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "(define (" & Func_Name & " " & Params & ")" & New_Line &
         "  " & Body_Code & ")" & New_Line & New_Line);
   end Emit_Define;

end Scheme_Emitter;
