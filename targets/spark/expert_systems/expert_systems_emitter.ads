--  STUNIR Expert Systems Emitter - Ada SPARK Specification
--  CLIPS and JESS emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Expert_Systems_Emitter is

   type Expert_System is (CLIPS, JESS);

   type Expert_Config is record
      System : Expert_System;
   end record;

   Default_Config : constant Expert_Config := (System => CLIPS);

   procedure Emit_Template (
      Template_Name : in Identifier_String;
      Slots         : in String;
      Content       : out Content_String;
      Config        : in Expert_Config;
      Status        : out Emitter_Status);

   procedure Emit_Rule (
      Rule_Name : in Identifier_String;
      LHS       : in String;
      RHS       : in String;
      Content   : out Content_String;
      Config    : in Expert_Config;
      Status    : out Emitter_Status);

end Expert_Systems_Emitter;
