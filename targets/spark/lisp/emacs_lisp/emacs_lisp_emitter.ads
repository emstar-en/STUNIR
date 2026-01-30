--  STUNIR Emacs Lisp Emitter - Ada SPARK Specification
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package Emacs_Lisp_Emitter is

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Status       : out Emitter_Status);

   procedure Emit_Defun (
      Func_Name : in Identifier_String;
      Params    : in String;
      Docstring : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status);

end Emacs_Lisp_Emitter;
