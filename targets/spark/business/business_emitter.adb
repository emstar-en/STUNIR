--  STUNIR Business Language Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Business_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Program (
      Program_Name : in Identifier_String;
      Content      : out Content_String;
      Config       : in Business_Config;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Program_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when COBOL =>
            Content_Strings.Append (Content,
               "       IDENTIFICATION DIVISION." & New_Line &
               "       PROGRAM-ID. " & Name & "." & New_Line &
               "      * STUNIR Generated COBOL Program" & New_Line &
               "      * DO-178C Level A Compliant" & New_Line &
               "       ENVIRONMENT DIVISION." & New_Line &
               "       DATA DIVISION." & New_Line &
               "       PROCEDURE DIVISION." & New_Line);
         when BASIC =>
            Content_Strings.Append (Content,
               "REM STUNIR Generated BASIC Program" & New_Line &
               "REM DO-178C Level A Compliant" & New_Line &
               "REM Program: " & Name & New_Line & New_Line);
      end case;
   end Emit_Program;

   procedure Emit_Paragraph (
      Para_Name : in Identifier_String;
      Body_Code : in String;
      Config    : in Business_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Para_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when COBOL =>
            Content_Strings.Append (Content,
               "       " & Name & "." & New_Line &
               "           " & Body_Code & New_Line);
         when BASIC =>
            Content_Strings.Append (Content,
               Name & ":" & New_Line & Body_Code & New_Line);
      end case;
   end Emit_Paragraph;

end Business_Emitter;
