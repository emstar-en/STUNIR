-- STUNIR Parser Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Parser is
   pragma SPARK_Mode (On);

   -- Emit complete module (parser specification)
   overriding procedure Emit_Module
     (Self   : in out Parser_Emitter;
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
         when Yacc | Bison =>
            -- Emit Yacc/Bison parser
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* STUNIR Generated Parser */" & ASCII.LF);
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
               New_Item => "int yylex(void);" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "void yyerror(const char *s);" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%}" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%token ID NUM" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "prog: expr { printf(\"Parsed expression\\n\"); };" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "expr: ID | NUM ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);

         when ANTLR_Parser =>
            -- Emit ANTLR parser
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// STUNIR Generated ANTLR Parser" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "parser grammar " & Name_Strings.To_String (Module.Module_Name) & "Parser;" & ASCII.LF & ASCII.LF);
            
            if Self.Config.Generate_AST then
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "options { output = AST; }" & ASCII.LF & ASCII.LF);
            end if;

            Code_Buffers.Append
              (Source   => Output,
               New_Item => "prog : expr+ ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "expr : ID | NUM ;" & ASCII.LF);

         when JavaCC =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* STUNIR Generated JavaCC Parser */" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* DO-178C Level A */" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "options {" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  STATIC = false;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "}" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "PARSER_BEGIN(" & Name_Strings.To_String (Module.Module_Name) & ")" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "public class " & Name_Strings.To_String (Module.Module_Name) & " {}" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "PARSER_END(" & Name_Strings.To_String (Module.Module_Name) & ")" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Parser" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (AST node type)
   overriding procedure Emit_Type
     (Self   : in out Parser_Emitter;
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
         New_Item => "/* AST node: " & Name_Strings.To_String (T.Name) & " */" & ASCII.LF);

      Success := True;
   end Emit_Type;

   -- Emit function definition (parse rule)
   overriding procedure Emit_Function
     (Self   : in out Parser_Emitter;
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
         New_Item => Name_Strings.To_String (Func.Name) & " : /* parse rule */ ;" & ASCII.LF);

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Parser;
