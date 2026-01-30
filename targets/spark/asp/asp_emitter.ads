--  STUNIR ASP Emitter - Ada SPARK Specification
--  Answer Set Programming (Clingo/DLV) emitter
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package ASP_Emitter is

   type ASP_Dialect is (Clingo, DLV);

   type ASP_Config is record
      Dialect : ASP_Dialect;
   end record;

   Default_Config : constant ASP_Config := (Dialect => Clingo);

   procedure Emit_Fact (
      Predicate : in Identifier_String;
      Args      : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status);

   procedure Emit_Rule (
      Head : in String;
      Body : in String;
      Content : out Content_String;
      Status : out Emitter_Status);

   procedure Emit_Constraint (
      Body    : in String;
      Content : out Content_String;
      Status  : out Emitter_Status);

end ASP_Emitter;
