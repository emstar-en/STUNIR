--  STUNIR Bytecode Emitter - Ada SPARK Specification
--  JVM/CLR bytecode emitter
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Bytecode_Emitter is

   type Bytecode_Target is (JVM, CLR, Python_Bytecode);

   type Bytecode_Config is record
      Target : Bytecode_Target;
   end record;

   Default_Config : constant Bytecode_Config := (Target => JVM);

   procedure Emit_Class (
      Class_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Bytecode_Config;
      Status     : out Emitter_Status);

   procedure Emit_Method (
      Method_Name : in Identifier_String;
      Descriptor  : in String;
      Content     : out Content_String;
      Config      : in Bytecode_Config;
      Status      : out Emitter_Status);

end Bytecode_Emitter;
