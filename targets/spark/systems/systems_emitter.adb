--  STUNIR Systems Language Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Systems_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Package (
      Package_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Systems_Config;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Package_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Ada_2012 | Ada_2022 =>
            Content_Strings.Append (Content,
               "--  STUNIR Generated Ada Package" & New_Line &
               "--  DO-178C Level A Compliant" & New_Line &
               "pragma SPARK_Mode (On);" & New_Line & New_Line &
               "package " & Name & " is" & New_Line & New_Line &
               "end " & Name & ";" & New_Line);
         when D_Language =>
            Content_Strings.Append (Content,
               "// STUNIR Generated D Module" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "module " & Name & ";" & New_Line & New_Line);
      end case;
   end Emit_Package;

   procedure Emit_Procedure (
      Proc_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Systems_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Proc_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Ada_2012 | Ada_2022 =>
            Content_Strings.Append (Content,
               "   procedure " & Name & " (" & Params & ") is" & New_Line &
               "   begin" & New_Line &
               "      " & Body_Code & New_Line &
               "   end " & Name & ";" & New_Line & New_Line);
         when D_Language =>
            Content_Strings.Append (Content,
               "void " & Name & "(" & Params & ") {" & New_Line &
               "    " & Body_Code & New_Line &
               "}" & New_Line & New_Line);
      end case;
   end Emit_Procedure;

end Systems_Emitter;
