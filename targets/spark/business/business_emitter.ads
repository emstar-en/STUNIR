--  STUNIR Business Language Emitter - Ada SPARK Specification
--  COBOL and BASIC emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Business_Emitter is

   type Business_Language is (COBOL, BASIC);

   type Business_Config is record
      Language : Business_Language;
   end record;

   Default_Config : constant Business_Config := (Language => COBOL);

   procedure Emit_Program (
      Program_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Business_Config;
      Status       : out Emitter_Status);

   procedure Emit_Paragraph (
      Para_Name : in Identifier_String;
      Body_Code : in String;
      Config    : in Business_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status);

end Business_Emitter;
