-- STUNIR Lexer Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Lexer is
   pragma SPARK_Mode (On);

   -- Emit complete module (lexer specification)
   overriding procedure Emit_Module
     (Self   : in out Lexer_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Generator is
         when Flex | Lex =>
            -- Emit Flex/Lex specification
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* STUNIR Generated Lexer */" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* DO-178C Level A */" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%{" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "#include <stdio.h>" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%}" & ASCII.LF & ASCII.LF);
            
            if Self.Config.Line_Tracking then
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "%option yylineno" & ASCII.LF);
            end if;

            Code_Buffers.Append
              (Source   => Output,
               New_Item => ASCII.LF & "%%" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "[a-zA-Z][a-zA-Z0-9]*  { return ID; }" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "[0-9]+                { return NUM; }" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "[ \t\n]+              { /* skip whitespace */ }" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ".                     { return yytext[0]; }" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);

         when JFlex =>
            -- Emit JFlex specification
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* STUNIR Generated JFlex Lexer */" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* DO-178C Level A */" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%class " & Name_Strings.To_String (Module.Module_Name) & "Lexer" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "[a-zA-Z][a-zA-Z0-9]* { return symbol(ID); }" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "[0-9]+               { return symbol(NUM); }" & ASCII.LF);

         when ANTLR_Lexer =>
            -- Emit ANTLR lexer rules
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// STUNIR Generated ANTLR Lexer" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "lexer grammar " & Name_Strings.To_String (Module.Module_Name) & "Lexer;" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "ID  : [a-zA-Z][a-zA-Z0-9]* ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "NUM : [0-9]+ ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "WS  : [ \t\r\n]+ -> skip ;" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Lexer" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (token type)
   overriding procedure Emit_Type
     (Self   : in out Lexer_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      Code_Buffers.Append
        (Source   => Output,
         New_Item => "/* Token type: " & Name_Strings.To_String (T.Name) & " */" & ASCII.LF);

      Success := True;
   end Emit_Type;

   -- Emit function definition (lexer action)
   overriding procedure Emit_Function
     (Self   : in out Lexer_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      Code_Buffers.Append
        (Source   => Output,
         New_Item => "/* Lexer action: " & Name_Strings.To_String (Func.Name) & " */" & ASCII.LF);

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Lexer;
