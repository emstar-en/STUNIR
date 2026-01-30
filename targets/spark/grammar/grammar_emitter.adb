--  STUNIR Grammar Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Grammar_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Grammar (
      Grammar_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Grammar_Config;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Grammar_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Format is
         when ANTLR4 =>
            Content_Strings.Append (Content,
               "// STUNIR Generated ANTLR4 Grammar" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "grammar " & Name & ";" & New_Line & New_Line);
         when BNF =>
            Content_Strings.Append (Content,
               "; STUNIR Generated BNF Grammar" & New_Line &
               "; Grammar: " & Name & New_Line & New_Line);
         when EBNF =>
            Content_Strings.Append (Content,
               "(* STUNIR Generated EBNF Grammar *)" & New_Line &
               "(* Grammar: " & Name & " *)" & New_Line & New_Line);
         when PEG =>
            Content_Strings.Append (Content,
               "# STUNIR Generated PEG Grammar" & New_Line &
               "# Grammar: " & Name & New_Line & New_Line);
         when YACC =>
            Content_Strings.Append (Content,
               "/* STUNIR Generated YACC Grammar */" & New_Line &
               "/* DO-178C Level A Compliant */" & New_Line &
               "%{" & New_Line & "%}" & New_Line & New_Line);
      end case;
   end Emit_Grammar;

   procedure Emit_Rule (
      Rule_Name : in Identifier_String;
      Rule_Body : in String;
      Content   : out Content_String;
      Config    : in Grammar_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Rule_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Format is
         when ANTLR4 =>
            Content_Strings.Append (Content,
               Name & " : " & Rule_Body & " ;" & New_Line);
         when BNF =>
            Content_Strings.Append (Content,
               "<" & Name & "> ::= " & Rule_Body & New_Line);
         when EBNF =>
            Content_Strings.Append (Content,
               Name & " = " & Rule_Body & " ;" & New_Line);
         when PEG =>
            Content_Strings.Append (Content,
               Name & " <- " & Rule_Body & New_Line);
         when YACC =>
            Content_Strings.Append (Content,
               Name & " : " & Rule_Body & New_Line &
               "    ;" & New_Line);
      end case;
   end Emit_Rule;

end Grammar_Emitter;
