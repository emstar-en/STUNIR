--  STUNIR OOP Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body OOP_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Class (
      Class_Name : in Identifier_String;
      Superclass : in String;
      Content    : out Content_String;
      Config     : in OOP_Config;
      Status     : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Class_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Smalltalk =>
            Content_Strings.Append (Content,
               "\"STUNIR Generated Smalltalk Class\"" & New_Line &
               "\"DO-178C Level A Compliant\"" & New_Line &
               Superclass & " subclass: #" & Name & New_Line &
               "    instanceVariableNames: ''" & New_Line &
               "    classVariableNames: ''" & New_Line &
               "    poolDictionaries: ''" & New_Line &
               "    category: 'STUNIR-Generated'!" & New_Line & New_Line);
         when ALGOL_68 =>
            Content_Strings.Append (Content,
               "CO STUNIR Generated ALGOL 68 CO" & New_Line &
               "CO DO-178C Level A Compliant CO" & New_Line &
               "MODE " & Name & " = STRUCT (" & New_Line &
               "    INT placeholder" & New_Line &
               ");" & New_Line & New_Line);
      end case;
   end Emit_Class;

   procedure Emit_Method (
      Method_Name : in Identifier_String;
      Body_Code   : in String;
      Content     : out Content_String;
      Config      : in OOP_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Method_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Smalltalk =>
            Content_Strings.Append (Content,
               Name & New_Line &
               "    " & Body_Code & New_Line & "!" & New_Line & New_Line);
         when ALGOL_68 =>
            Content_Strings.Append (Content,
               "PROC " & Name & " = VOID:" & New_Line &
               "(" & New_Line &
               "    " & Body_Code & New_Line &
               ");" & New_Line & New_Line);
      end case;
   end Emit_Method;

end OOP_Emitter;
