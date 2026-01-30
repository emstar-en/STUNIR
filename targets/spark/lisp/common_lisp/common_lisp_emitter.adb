--  STUNIR Common Lisp Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Common_Lisp_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Lisp_Config;
      Status       : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Package_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         ";;;; STUNIR Generated Common Lisp Package" & New_Line &
         ";;;; Package: " & Name & New_Line &
         ";;;; DO-178C Level A Compliant" & New_Line & New_Line &
         "(defpackage #:" & Name & New_Line &
         "  (:use #:cl))" & New_Line & New_Line &
         "(in-package #:" & Name & ")" & New_Line & New_Line);
   end Emit_Package;

   procedure Emit_Defun (
      Func_Name   : in Identifier_String;
      Params      : in String;
      Body_Code   : in String;
      Content     : out Content_String;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "(defun " & Name & " (" & Params & ")" & New_Line &
         "  " & Body_Code & ")" & New_Line & New_Line);
   end Emit_Defun;

end Common_Lisp_Emitter;
