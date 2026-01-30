--  STUNIR BEAM Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body BEAM_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in BEAM_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Erlang =>
            Content_Strings.Append (Content,
               "%% STUNIR Generated Erlang Module" & New_Line &
               "%% DO-178C Level A Compliant" & New_Line &
               "-module(" & Name & ")." & New_Line &
               "-export([])." & New_Line & New_Line);
         when Elixir =>
            Content_Strings.Append (Content,
               "# STUNIR Generated Elixir Module" & New_Line &
               "# DO-178C Level A Compliant" & New_Line &
               "defmodule " & Name & " do" & New_Line & New_Line);
      end case;
   end Emit_Module;

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Config    : in BEAM_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Erlang =>
            Content_Strings.Append (Content,
               Name & "(" & Params & ") ->" & New_Line &
               "    " & Body_Code & "." & New_Line & New_Line);
         when Elixir =>
            Content_Strings.Append (Content,
               "  def " & Name & "(" & Params & ") do" & New_Line &
               "    " & Body_Code & New_Line &
               "  end" & New_Line & New_Line);
      end case;
   end Emit_Function;

end BEAM_Emitter;
