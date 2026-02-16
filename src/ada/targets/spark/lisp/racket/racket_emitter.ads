--  STUNIR Racket Emitter - Ada SPARK Specification
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Racket_Emitter is

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Language    : in String;
      Content     : out Content_String;
      Status      : out Emitter_Status);

   procedure Emit_Define (
      Name      : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status);

end Racket_Emitter;
