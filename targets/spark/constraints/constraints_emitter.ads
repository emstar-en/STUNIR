--  STUNIR Constraints Emitter - Ada SPARK Specification
--  MiniZinc and CHR emitters
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Constraints_Emitter is

   type Constraint_Language is (MiniZinc, CHR);

   type Constraints_Config is record
      Language : Constraint_Language;
   end record;

   Default_Config : constant Constraints_Config := (Language => MiniZinc);

   procedure Emit_Model (
      Model_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Constraints_Config;
      Status     : out Emitter_Status);

   procedure Emit_Variable (
      Var_Name : in Identifier_String;
      Domain   : in String;
      Content  : out Content_String;
      Config   : in Constraints_Config;
      Status   : out Emitter_Status);

   procedure Emit_Constraint (
      Constraint_Expr : in String;
      Content         : out Content_String;
      Config          : in Constraints_Config;
      Status          : out Emitter_Status);

end Constraints_Emitter;
