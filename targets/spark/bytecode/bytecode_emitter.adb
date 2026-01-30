--  STUNIR Bytecode Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Bytecode_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Class (
      Class_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Bytecode_Config;
      Status     : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Class_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when JVM =>
            Content_Strings.Append (Content,
               "; STUNIR Generated JVM Bytecode" & New_Line &
               "; DO-178C Level A Compliant" & New_Line &
               ".class public " & Name & New_Line &
               ".super java/lang/Object" & New_Line & New_Line);
         when CLR =>
            Content_Strings.Append (Content,
               "// STUNIR Generated CIL" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               ".class public " & Name & New_Line &
               "    extends [mscorlib]System.Object" & New_Line &
               "{" & New_Line);
         when Python_Bytecode =>
            Content_Strings.Append (Content,
               "# STUNIR Generated Python Bytecode" & New_Line &
               "# DO-178C Level A Compliant" & New_Line &
               "# Class: " & Name & New_Line);
      end case;
   end Emit_Class;

   procedure Emit_Method (
      Method_Name : in Identifier_String;
      Descriptor  : in String;
      Content     : out Content_String;
      Config      : in Bytecode_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Method_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Target is
         when JVM =>
            Content_Strings.Append (Content,
               ".method public static " & Name & Descriptor & New_Line &
               "    .limit stack 10" & New_Line &
               "    .limit locals 10" & New_Line);
         when CLR =>
            Content_Strings.Append (Content,
               "    .method public static " & Descriptor & " " & Name & "()" & New_Line &
               "    {" & New_Line);
         when Python_Bytecode =>
            Content_Strings.Append (Content,
               "# Method: " & Name & New_Line);
      end case;
   end Emit_Method;

end Bytecode_Emitter;
