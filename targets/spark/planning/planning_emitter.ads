--  STUNIR Planning Emitter - Ada SPARK Specification
--  PDDL emitter for AI planning
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Planning_Emitter is

   type PDDL_Version is (PDDL_2_1, PDDL_3_0, PDDL_3_1);

   type Planning_Config is record
      Version : PDDL_Version;
   end record;

   Default_Config : constant Planning_Config := (Version => PDDL_3_1);

   procedure Emit_Domain (
      Domain_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Planning_Config;
      Status      : out Emitter_Status);

   procedure Emit_Action (
      Action_Name   : in Identifier_String;
      Parameters    : in String;
      Precondition  : in String;
      Effect        : in String;
      Content       : out Content_String;
      Status        : out Emitter_Status);

   procedure Emit_Problem (
      Problem_Name : in Identifier_String;
      Domain_Name  : in String;
      Content      : out Content_String;
      Status       : out Emitter_Status);

end Planning_Emitter;
