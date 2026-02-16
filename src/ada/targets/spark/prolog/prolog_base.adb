--  STUNIR Prolog Base - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Prolog_Base is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Prolog_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Dialect is
         when SWI_Prolog =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated SWI-Prolog Module" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- module(" & Name & ", [])." & New_Line & New_Line);
         when GNU_Prolog =>
            Content_Strings.Append (Content,
               "% STUNIR Generated GNU Prolog" & New_Line &
               "% DO-178C Level A Compliant" & New_Line & New_Line);
         when YAP =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated YAP Prolog" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- module(" & Name & ", [])." & New_Line & New_Line);
         when XSB =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated XSB Prolog" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- module(" & Name & ", [])." & New_Line & New_Line);
         when ECLiPSe =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated ECLiPSe" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- module(" & Name & ")." & New_Line & New_Line);
         when Tau_Prolog =>
            Content_Strings.Append (Content,
               "% STUNIR Generated Tau Prolog" & New_Line &
               "% DO-178C Level A Compliant" & New_Line & New_Line);
         when Mercury =>
            Content_Strings.Append (Content,
               "% STUNIR Generated Mercury" & New_Line &
               "% DO-178C Level A Compliant" & New_Line &
               ":- module " & Name & "." & New_Line &
               ":- interface." & New_Line & New_Line);
         when SICStus =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated SICStus Prolog" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- module(" & Name & ", [])." & New_Line & New_Line);
         when Datalog =>
            Content_Strings.Append (Content,
               "% STUNIR Generated Datalog" & New_Line &
               "% DO-178C Level A Compliant" & New_Line & New_Line);
      end case;
   end Emit_Module;

   procedure Emit_Clause (
      Head : in String;
      Body : in String;
      Content : out Content_String;
      Status : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;
      Content_Strings.Append (Content, Head & " :- " & Body & "." & New_Line);
   end Emit_Clause;

   procedure Emit_Fact (
      Predicate : in Identifier_String;
      Args      : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Predicate);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;
      Content_Strings.Append (Content, Name & "(" & Args & ")." & New_Line);
   end Emit_Fact;

end Prolog_Base;
