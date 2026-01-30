--  STUNIR Grammar Emitter - Ada SPARK Specification
--  ANTLR, BNF, EBNF, PEG, YACC emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Grammar_Emitter is

   type Grammar_Format is (ANTLR4, BNF, EBNF, PEG, YACC);

   type Grammar_Config is record
      Format : Grammar_Format;
   end record;

   Default_Config : constant Grammar_Config := (Format => ANTLR4);

   procedure Emit_Grammar (
      Grammar_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Grammar_Config;
      Status       : out Emitter_Status);

   procedure Emit_Rule (
      Rule_Name : in Identifier_String;
      Rule_Body : in String;
      Content   : out Content_String;
      Config    : in Grammar_Config;
      Status    : out Emitter_Status);

end Grammar_Emitter;
