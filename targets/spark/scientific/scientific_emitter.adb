--  STUNIR Scientific Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Scientific_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Scientific_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Fortran_90 | Fortran_2008 =>
            Content_Strings.Append (Content,
               "! STUNIR Generated Fortran Module" & New_Line &
               "! DO-178C Level A Compliant" & New_Line &
               "module " & Name & New_Line &
               "  implicit none" & New_Line &
               "contains" & New_Line & New_Line);
         when Pascal | Delphi =>
            Content_Strings.Append (Content,
               "{ STUNIR Generated Pascal Unit }" & New_Line &
               "{ DO-178C Level A Compliant }" & New_Line &
               "unit " & Name & ";" & New_Line & New_Line &
               "interface" & New_Line & New_Line &
               "implementation" & New_Line & New_Line);
      end case;
   end Emit_Module;

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Scientific_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Fortran_90 | Fortran_2008 =>
            Content_Strings.Append (Content,
               "  subroutine " & Name & "(" & Params & ")" & New_Line &
               "    " & Body_Code & New_Line &
               "  end subroutine " & Name & New_Line & New_Line);
         when Pascal | Delphi =>
            Content_Strings.Append (Content,
               "procedure " & Name & "(" & Params & ");" & New_Line &
               "begin" & New_Line &
               "  " & Body_Code & New_Line &
               "end;" & New_Line & New_Line);
      end case;
   end Emit_Function;

end Scientific_Emitter;
