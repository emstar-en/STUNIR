--  STUNIR Systems Language Emitter - Ada SPARK Specification
--  Ada and D emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Systems_Emitter is

   type Systems_Language is (Ada_2012, Ada_2022, D_Language);

   type Systems_Config is record
      Language : Systems_Language;
   end record;

   Default_Config : constant Systems_Config := (Language => Ada_2012);

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Systems_Config;
      Status       : out Emitter_Status);

   procedure Emit_Procedure (
      Proc_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Systems_Config;
      Status    : out Emitter_Status);

end Systems_Emitter;
