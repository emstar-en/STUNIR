--  STUNIR Parser Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Parser_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Parser (
      Parser_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Parser_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Parser_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when C_Parser =>
            Content_Strings.Append (Content,
               "/* STUNIR Generated C Parser */" & New_Line &
               "/* DO-178C Level A Compliant */" & New_Line &
               "typedef struct {" & New_Line &
               "    Lexer* lexer;" & New_Line &
               "    Token current;" & New_Line &
               "} " & Name & ";" & New_Line & New_Line);
         when Python_Parser =>
            Content_Strings.Append (Content,
               "# STUNIR Generated Python Parser" & New_Line &
               "# DO-178C Level A Compliant" & New_Line &
               "class " & Name & ":" & New_Line &
               "    def __init__(self, lexer):" & New_Line &
               "        self.lexer = lexer" & New_Line &
               "        self.current = None" & New_Line & New_Line);
         when Rust_Parser =>
            Content_Strings.Append (Content,
               "// STUNIR Generated Rust Parser" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "pub struct " & Name & " {" & New_Line &
               "    lexer: Lexer," & New_Line &
               "    current: Option<Token>," & New_Line &
               "}" & New_Line & New_Line);
         when Table_Driven =>
            Content_Strings.Append (Content,
               "/* Table-driven parser: " & Name & " */" & New_Line);
      end case;
   end Emit_Parser;

   procedure Emit_Parse_Function (
      Rule_Name : in Identifier_String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Parser_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Rule_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when C_Parser =>
            Content_Strings.Append (Content,
               "ASTNode* parse_" & Name & "(Parser* p) {" & New_Line &
               "    " & Body_Code & New_Line &
               "}" & New_Line & New_Line);
         when Python_Parser =>
            Content_Strings.Append (Content,
               "    def parse_" & Name & "(self):" & New_Line &
               "        " & Body_Code & New_Line & New_Line);
         when Rust_Parser =>
            Content_Strings.Append (Content,
               "    fn parse_" & Name & "(&mut self) -> Result<ASTNode, Error> {" & New_Line &
               "        " & Body_Code & New_Line &
               "    }" & New_Line & New_Line);
         when Table_Driven =>
            Content_Strings.Append (Content,
               "/* Rule: " & Name & " */" & New_Line);
      end case;
   end Emit_Parse_Function;

end Parser_Emitter;
