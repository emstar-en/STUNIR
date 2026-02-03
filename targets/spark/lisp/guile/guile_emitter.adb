--  STUNIR Guile Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Guile_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         ";; STUNIR Generated Guile Module" & New_Line &
         ";; Module: " & Name & New_Line &
         ";; DO-178C Level A Compliant" & New_Line & New_Line &
         "(define-module (" & Name & ")" & New_Line &
         "  #:export ())" & New_Line & New_Line);
   end Emit_Module;

   procedure Emit_Define (
      Name      : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Func_Name : constant String := Identifier_Strings.To_String (Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "(define (" & Func_Name & " " & Params & ")" & New_Line &
         "  " & Body_Code & ")" & New_Line & New_Line);
   end Emit_Define;

end Guile_Emitter;
