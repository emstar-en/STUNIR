--  STUNIR OOP Emitter - Ada SPARK Specification
--  Smalltalk and ALGOL emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package OOP_Emitter is

   type OOP_Language is (Smalltalk, ALGOL_68);

   type OOP_Config is record
      Language : OOP_Language;
   end record;

   Default_Config : constant OOP_Config := (Language => Smalltalk);

   procedure Emit_Class (
      Class_Name : in Identifier_String;
      Superclass : in String;
      Content    : out Content_String;
      Config     : in OOP_Config;
      Status     : out Emitter_Status);

   procedure Emit_Method (
      Method_Name : in Identifier_String;
      Body_Code   : in String;
      Content     : out Content_String;
      Config      : in OOP_Config;
      Status      : out Emitter_Status);

end OOP_Emitter;
