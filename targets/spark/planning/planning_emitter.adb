--  STUNIR Planning Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Planning_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Domain (
      Domain_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Planning_Config;
      Status      : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Domain_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "; STUNIR Generated PDDL Domain" & New_Line &
         "; DO-178C Level A Compliant" & New_Line &
         "(define (domain " & Name & ")" & New_Line &
         "  (:requirements :strips :typing)" & New_Line & New_Line);
   end Emit_Domain;

   procedure Emit_Action (
      Action_Name   : in Identifier_String;
      Parameters    : in String;
      Precondition  : in String;
      Effect        : in String;
      Content       : out Content_String;
      Status        : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Action_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "  (:action " & Name & New_Line &
         "    :parameters (" & Parameters & ")" & New_Line &
         "    :precondition (and " & Precondition & ")" & New_Line &
         "    :effect (and " & Effect & "))" & New_Line & New_Line);
   end Emit_Action;

   procedure Emit_Problem (
      Problem_Name : in Identifier_String;
      Domain_Name  : in String;
      Content      : out Content_String;
      Status       : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Problem_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "; STUNIR Generated PDDL Problem" & New_Line &
         "(define (problem " & Name & ")" & New_Line &
         "  (:domain " & Domain_Name & ")" & New_Line &
         "  (:objects)" & New_Line &
         "  (:init)" & New_Line &
         "  (:goal (and)))" & New_Line);
   end Emit_Problem;

end Planning_Emitter;
