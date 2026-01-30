--  STUNIR Lexer Emitter - Ada SPARK Specification
--  Lexer code generation for C, Python, Rust
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Lexer_Emitter is

   type Lexer_Target is (C_Lexer, Python_Lexer, Rust_Lexer, Table_Driven);

   type Lexer_Config is record
      Target : Lexer_Target;
   end record;

   Default_Config : constant Lexer_Config := (Target => C_Lexer);

   procedure Emit_Token_Enum (
      Tokens  : in String;
      Content : out Content_String;
      Config  : in Lexer_Config;
      Status  : out Emitter_Status);

   procedure Emit_Lexer (
      Lexer_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Lexer_Config;
      Status     : out Emitter_Status);

end Lexer_Emitter;
