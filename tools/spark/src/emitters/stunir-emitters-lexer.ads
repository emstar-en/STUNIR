-- STUNIR Lexer Emitter (SPARK Specification)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters
-- Support: Flex, Lex, JFlex, ANTLR Lexer

with STUNIR.Semantic_IR; use STUNIR.Semantic_IR;
with STUNIR.Emitters;    use STUNIR.Emitters;

package STUNIR.Emitters.Lexer is
   pragma SPARK_Mode (On);
   pragma Elaborate_Body;

   -- Lexer generator enumeration
   type Lexer_Generator is
     (Flex,
      Lex,
      JFlex,
      ANTLR_Lexer,
      RE2C,
      Ragel,
      Hand_Written);

   -- Lexer emitter configuration
   type Lexer_Config is record
      Generator      : Lexer_Generator := Flex;
      Case_Sensitive : Boolean := True;
      Unicode_Support: Boolean := False;
      Line_Tracking  : Boolean := True;
      Indent_Size    : Positive := 2;
      Max_Line_Width : Positive := 100;
   end record;

   -- Default configuration
   Default_Config : constant Lexer_Config :=
     (Generator      => Flex,
      Case_Sensitive => True,
      Unicode_Support=> False,
      Line_Tracking  => True,
      Indent_Size    => 2,
      Max_Line_Width => 100);

   -- Lexer emitter type
   type Lexer_Emitter is new Base_Emitter with record
      Config : Lexer_Config := Default_Config;
   end record;

   -- Override abstract methods from Base_Emitter
   overriding procedure Emit_Module
     (Self   : in out Lexer_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Is_Valid_Module (Module),
     Post'Class => (if Success then Code_Buffers.Length (Output) > 0);

   overriding procedure Emit_Type
     (Self   : in out Lexer_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => T.Field_Cnt > 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

   overriding procedure Emit_Function
     (Self   : in out Lexer_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   with
     Pre'Class  => Func.Arg_Cnt >= 0,
     Post'Class => (if Success then Code_Buffers.Length (Output) >= 0);

end STUNIR.Emitters.Lexer;
