--  STUNIR Parser Emitter - Ada SPARK Specification
--  Parser code generation for C, Python, Rust
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Parser_Emitter is

   type Parser_Target is (C_Parser, Python_Parser, Rust_Parser, Table_Driven);
   type Parser_Type is (LL1, LR1, LALR, Recursive_Descent);

   type Parser_Config is record
      Target      : Parser_Target;
      Parser_Kind : Parser_Type;
   end record;

   Default_Config : constant Parser_Config := (
      Target => C_Parser,
      Parser_Kind => Recursive_Descent
   );

   procedure Emit_Parser (
      Parser_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Parser_Config;
      Status      : out Emitter_Status);

   procedure Emit_Parse_Function (
      Rule_Name : in Identifier_String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Parser_Config;
      Status    : out Emitter_Status);

end Parser_Emitter;
