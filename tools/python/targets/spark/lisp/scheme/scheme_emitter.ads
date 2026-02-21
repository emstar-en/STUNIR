--  STUNIR Scheme Emitter - Ada SPARK Specification
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;
with Lisp_Base; use Lisp_Base;

package Scheme_Emitter is

   type Scheme_Standard is (R5RS, R6RS, R7RS);

   type Scheme_Config is record
      Standard : Scheme_Standard;
   end record;

   Default_Scheme_Config : constant Scheme_Config := (
      Standard => R7RS
   );

   procedure Emit_Library (
      Library_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Scheme_Config;
      Status       : out Emitter_Status);

   procedure Emit_Define (
      Name    : in Identifier_String;
      Params  : in String;
      Body_Code : in String;
      Content : out Content_String;
      Status  : out Emitter_Status);

end Scheme_Emitter;
