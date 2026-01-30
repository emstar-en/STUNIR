--  STUNIR Expert Systems Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Expert_Systems_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Template (
      Template_Name : in Identifier_String;
      Slots         : in String;
      Content       : out Content_String;
      Config        : in Expert_Config;
      Status        : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Template_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.System is
         when CLIPS =>
            Content_Strings.Append (Content,
               "; STUNIR Generated CLIPS Template" & New_Line &
               "(deftemplate " & Name & New_Line &
               "   " & Slots & ")" & New_Line & New_Line);
         when JESS =>
            Content_Strings.Append (Content,
               "; STUNIR Generated JESS Template" & New_Line &
               "(deftemplate " & Name & New_Line &
               "   " & Slots & ")" & New_Line & New_Line);
      end case;
   end Emit_Template;

   procedure Emit_Rule (
      Rule_Name : in Identifier_String;
      LHS       : in String;
      RHS       : in String;
      Content   : out Content_String;
      Config    : in Expert_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Rule_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.System is
         when CLIPS =>
            Content_Strings.Append (Content,
               "(defrule " & Name & New_Line &
               "   " & LHS & New_Line &
               "   =>" & New_Line &
               "   " & RHS & ")" & New_Line & New_Line);
         when JESS =>
            Content_Strings.Append (Content,
               "(defrule " & Name & New_Line &
               "   " & LHS & New_Line &
               "   =>" & New_Line &
               "   " & RHS & ")" & New_Line & New_Line);
      end case;
   end Emit_Rule;

end Expert_Systems_Emitter;
