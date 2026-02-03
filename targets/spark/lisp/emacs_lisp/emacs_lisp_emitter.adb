--  STUNIR Emacs Lisp Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Emacs_Lisp_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Package_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         ";;; " & Name & ".el --- STUNIR Generated -*- lexical-binding: t -*-" & New_Line &
         ";;; DO-178C Level A Compliant" & New_Line & New_Line &
         ";;; Commentary:" & New_Line &
         ";;; STUNIR generated Emacs Lisp package" & New_Line & New_Line &
         ";;; Code:" & New_Line & New_Line);
   end Emit_Package;

   procedure Emit_Defun (
      Func_Name : in Identifier_String;
      Params    : in String;
      Docstring : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "(defun " & Name & " (" & Params & ")" & New_Line &
         "  \"" & Docstring & "\"" & New_Line &
         "  " & Body_Code & ")" & New_Line & New_Line);
   end Emit_Defun;

end Emacs_Lisp_Emitter;
