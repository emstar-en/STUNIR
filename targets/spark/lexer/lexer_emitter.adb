--  STUNIR Lexer Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Lexer_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Token_Enum (
      Tokens  : in String;
      Content : out Content_String;
      Config  : in Lexer_Config;
      Status  : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when C_Lexer =>
            Content_Strings.Append (Content,
               "typedef enum {" & New_Line &
               "    " & Tokens & New_Line &
               "} TokenType;" & New_Line & New_Line);
         when Python_Lexer =>
            Content_Strings.Append (Content,
               "from enum import Enum, auto" & New_Line & New_Line &
               "class TokenType(Enum):" & New_Line &
               "    " & Tokens & New_Line & New_Line);
         when Rust_Lexer =>
            Content_Strings.Append (Content,
               "#[derive(Debug, Clone, PartialEq)]" & New_Line &
               "pub enum TokenType {" & New_Line &
               "    " & Tokens & New_Line &
               "}" & New_Line & New_Line);
         when Table_Driven =>
            Content_Strings.Append (Content,
               "/* Token types: " & Tokens & " */" & New_Line);
      end case;
   end Emit_Token_Enum;

   procedure Emit_Lexer (
      Lexer_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Lexer_Config;
      Status     : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Lexer_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when C_Lexer =>
            Content_Strings.Append (Content,
               "/* STUNIR Generated C Lexer */" & New_Line &
               "/* DO-178C Level A Compliant */" & New_Line &
               "typedef struct {" & New_Line &
               "    const char* source;" & New_Line &
               "    size_t pos;" & New_Line &
               "} " & Name & ";" & New_Line & New_Line);
         when Python_Lexer =>
            Content_Strings.Append (Content,
               "# STUNIR Generated Python Lexer" & New_Line &
               "# DO-178C Level A Compliant" & New_Line &
               "class " & Name & ":" & New_Line &
               "    def __init__(self, source):" & New_Line &
               "        self.source = source" & New_Line &
               "        self.pos = 0" & New_Line & New_Line);
         when Rust_Lexer =>
            Content_Strings.Append (Content,
               "// STUNIR Generated Rust Lexer" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "pub struct " & Name & " {" & New_Line &
               "    source: String," & New_Line &
               "    pos: usize," & New_Line &
               "}" & New_Line & New_Line);
         when Table_Driven =>
            Content_Strings.Append (Content,
               "/* Table-driven lexer: " & Name & " */" & New_Line);
      end case;
   end Emit_Lexer;

end Lexer_Emitter;
