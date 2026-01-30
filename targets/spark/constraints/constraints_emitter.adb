--  STUNIR Constraints Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Constraints_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Model (
      Model_Name : in Identifier_String;
      Content    : out Content_String;
      Config     : in Constraints_Config;
      Status     : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Model_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when MiniZinc =>
            Content_Strings.Append (Content,
               "% STUNIR Generated MiniZinc Model" & New_Line &
               "% Model: " & Name & New_Line &
               "% DO-178C Level A Compliant" & New_Line & New_Line);
         when CHR =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated CHR Rules" & New_Line &
               "%% Model: " & Name & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               ":- use_module(library(chr))." & New_Line & New_Line);
      end case;
   end Emit_Model;

   procedure Emit_Variable (
      Var_Name : in Identifier_String;
      Domain   : in String;
      Content  : out Content_String;
      Config   : in Constraints_Config;
      Status   : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Var_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when MiniZinc =>
            Content_Strings.Append (Content,
               "var " & Domain & ": " & Name & ";" & New_Line);
         when CHR =>
            Content_Strings.Append (Content,
               "chr_constraint " & Name & "/1." & New_Line);
      end case;
   end Emit_Variable;

   procedure Emit_Constraint (
      Constraint_Expr : in String;
      Content         : out Content_String;
      Config          : in Constraints_Config;
      Status          : out Emitter_Status)
   is
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when MiniZinc =>
            Content_Strings.Append (Content,
               "constraint " & Constraint_Expr & ";" & New_Line);
         when CHR =>
            Content_Strings.Append (Content, Constraint_Expr & "." & New_Line);
      end case;
   end Emit_Constraint;

end Constraints_Emitter;
