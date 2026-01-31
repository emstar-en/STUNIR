-- STUNIR BEAM VM Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.BEAM is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out BEAM_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Erlang =>
            Code_Buffers.Append (Source => Output, New_Item => "% STUNIR Generated Erlang" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "% DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "-module(" & Name_Strings.To_String (Module.Module_Name) & ")." & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "-export([])." & ASCII.LF);
            if Self.Config.Use_OTP then
               Code_Buffers.Append (Source => Output, New_Item => "-behaviour(gen_server)." & ASCII.LF);
            end if;
         when Elixir =>
            Code_Buffers.Append (Source => Output, New_Item => "# STUNIR Generated Elixir" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "defmodule " & Name_Strings.To_String (Module.Module_Name) & " do" & ASCII.LF);
            if Self.Config.Use_OTP then
               Code_Buffers.Append (Source => Output, New_Item => "  use GenServer" & ASCII.LF);
            end if;
            Code_Buffers.Append (Source => Output, New_Item => "end" & ASCII.LF);
         when LFE =>
            Code_Buffers.Append (Source => Output, New_Item => ";;; STUNIR Generated LFE" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "(defmodule " & Name_Strings.To_String (Module.Module_Name) & ")" & ASCII.LF);
         when Gleam =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Gleam" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out BEAM_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Erlang =>
            Code_Buffers.Append (Source => Output, New_Item => "-record(" & Name_Strings.To_String (T.Name) & ", {");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (T.Fields (I).Name));
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => "})." & ASCII.LF);
         when Elixir =>
            Code_Buffers.Append (Source => Output, New_Item => "  defstruct [");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (T.Fields (I).Name));
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => "]" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "% Type: " & Name_Strings.To_String (T.Name) & ASCII.LF);
      end case;
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out BEAM_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Language is
         when Erlang =>
            Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (Func.Name) & "(");
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (Func.Args (I).Name));
               if I < Func.Arg_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => ") -> ok." & ASCII.LF);
         when Elixir =>
            Code_Buffers.Append (Source => Output, New_Item => "  def " & Name_Strings.To_String (Func.Name) & "(");
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => Name_Strings.To_String (Func.Args (I).Name));
               if I < Func.Arg_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append (Source => Output, New_Item => ") do" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "    :ok" & ASCII.LF);
            Code_Buffers.Append (Source => Output, New_Item => "  end" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "% Function: " & Name_Strings.To_String (Func.Name) & ASCII.LF);
      end case;
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.BEAM;
