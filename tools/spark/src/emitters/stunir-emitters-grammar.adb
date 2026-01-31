-- STUNIR Grammar Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Grammar is
   pragma SPARK_Mode (On);

   -- Get file extension for grammar language
   function Get_Grammar_Extension (Lang : Grammar_Language) return String is
   begin
      case Lang is
         when ANTLR_v4 | ANTLR_v3 => return ".g4";
         when PEG => return ".peg";
         when BNF => return ".bnf";
         when EBNF => return ".ebnf";
         when Yacc | Bison => return ".y";
         when LALR | LL_Star => return ".grammar";
      end case;
   end Get_Grammar_Extension;

   -- Emit complete module (grammar specification)
   overriding procedure Emit_Module
     (Self   : in out Grammar_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Language is
         when ANTLR_v4 | ANTLR_v3 =>
            -- Emit ANTLR grammar
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// STUNIR Generated ANTLR Grammar" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "grammar " & Name_Strings.To_String (Module.Module_Name) & ";" & ASCII.LF & ASCII.LF);
            
            if Self.Config.Generate_AST then
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "options {" & ASCII.LF);
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "  output = AST;" & ASCII.LF);
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "}" & ASCII.LF & ASCII.LF);
            end if;

            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// Parser rules" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "prog : expr+ ;" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// Lexer rules" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "ID : [a-zA-Z]+ ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "WS : [ \t\r\n]+ -> skip ;" & ASCII.LF);

         when PEG =>
            -- Emit PEG grammar
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated PEG Grammar" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => Name_Strings.To_String (Module.Module_Name) & " <-" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  Spacing (Expr Spacing)+" & ASCII.LF);

         when BNF | EBNF =>
            -- Emit BNF/EBNF grammar
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(* STUNIR Generated BNF Grammar *)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(* DO-178C Level A *)" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "<prog> ::= <expr>+" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "<expr> ::= <term> | <term> '+' <expr>" & ASCII.LF);

         when Yacc | Bison =>
            -- Emit Yacc/Bison grammar
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "/* STUNIR Generated Yacc Grammar */" & ASCII.LF);
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
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "prog: expr ;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "%%" & ASCII.LF);

         when others =>
            -- Generic grammar format
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Grammar" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (grammar rule)
   overriding procedure Emit_Type
     (Self   : in out Grammar_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Language is
         when ANTLR_v4 | ANTLR_v3 =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => Name_Strings.To_String (T.Name) & " : ");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (T.Fields (I).Name));
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => " ");
               end if;
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => " ;" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "<" & Name_Strings.To_String (T.Name) & "> ::= ... ;" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Type;

   -- Emit function definition (grammar production rule)
   overriding procedure Emit_Function
     (Self   : in out Grammar_Emitter;
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
         New_Item => Name_Strings.To_String (Func.Name) & " : /* production rule */ ;" & ASCII.LF);

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Grammar;
