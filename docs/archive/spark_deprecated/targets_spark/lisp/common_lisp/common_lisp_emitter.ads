--  STUNIR Common Lisp Emitter - Ada SPARK Specification
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;
with Lisp_Base; use Lisp_Base;

package Common_Lisp_Emitter is

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Lisp_Config;
      Status       : out Emitter_Status);

   procedure Emit_Defun (
      Func_Name   : in Identifier_String;
      Params      : in String;
      Body_Code   : in String;
      Content     : out Content_String;
      Status      : out Emitter_Status);

end Common_Lisp_Emitter;
