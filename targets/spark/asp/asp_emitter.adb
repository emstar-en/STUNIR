--  STUNIR ASP Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body ASP_Emitter is

   New_Line : constant Character := ASCII.LF;

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

   procedure Emit_Rule (
      Head : in String;
      Body : in String;
      Content : out Content_String;
      Status : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;
      Content_Strings.Append (Content, Head & " :- " & Body & "." & New_Line);
   end Emit_Rule;

   procedure Emit_Constraint (
      Body    : in String;
      Content : out Content_String;
      Status  : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;
      Content_Strings.Append (Content, ":- " & Body & "." & New_Line);
   end Emit_Constraint;

end ASP_Emitter;
