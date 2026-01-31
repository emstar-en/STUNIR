-- STUNIR Functional Programming Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Functional is
   pragma SPARK_Mode (On);

   -- Map IR primitive types to functional language types
   function Get_FP_Type (Prim : IR_Primitive_Type; Lang : Functional_Language) return String is
   begin
      case Lang is
         when Haskell =>
            case Prim is
               when Type_String => return "String";
               when Type_Int | Type_I32 => return "Int";
               when Type_I8 | Type_I16 => return "Int";
               when Type_I64 => return "Integer";
               when Type_U8 | Type_U16 | Type_U32 => return "Word";
               when Type_U64 => return "Word64";
               when Type_Float | Type_F32 => return "Float";
               when Type_F64 => return "Double";
               when Type_Bool => return "Bool";
               when Type_Void => return "()";
            end case;

         when OCaml | F_Sharp =>
            case Prim is
               when Type_String => return "string";
               when Type_Int | Type_I32 => return "int";
               when Type_I8 | Type_I16 | Type_I64 => return "int";
               when Type_U8 | Type_U16 | Type_U32 | Type_U64 => return "int";
               when Type_Float | Type_F32 | Type_F64 => return "float";
               when Type_Bool => return "bool";
               when Type_Void => return "unit";
            end case;

         when others =>
            case Prim is
               when Type_String => return "String";
               when Type_Int | Type_I32 => return "Integer";
               when Type_I8 | Type_I16 | Type_I64 => return "Integer";
               when Type_U8 | Type_U16 | Type_U32 | Type_U64 => return "Integer";
               when Type_Float | Type_F32 | Type_F64 => return "Float";
               when Type_Bool => return "Boolean";
               when Type_Void => return "Unit";
            end case;
      end case;
   end Get_FP_Type;

   -- Emit complete module
   overriding procedure Emit_Module
     (Self   : in out Functional_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Language is
         when Haskell =>
            -- Emit Haskell module
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-- STUNIR Generated Haskell" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-- DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "module " & Name_Strings.To_String (Module.Module_Name) & " where" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-- | STUNIR generated pure functional code" & ASCII.LF);

         when OCaml =>
            -- Emit OCaml module
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(* STUNIR Generated OCaml *)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(* DO-178C Level A *)" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "module " & Name_Strings.To_String (Module.Module_Name) & " = struct" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (* STUNIR generated code *)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "end" & ASCII.LF);

         when F_Sharp =>
            -- Emit F# module
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// STUNIR Generated F#" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "module " & Name_Strings.To_String (Module.Module_Name) & ASCII.LF & ASCII.LF);

         when Erlang =>
            -- Emit Erlang module
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% STUNIR Generated Erlang" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-module(" & Name_Strings.To_String (Module.Module_Name) & ")." & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-export([])." & ASCII.LF);

         when Elixir =>
            -- Emit Elixir module
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Elixir" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "defmodule " & Name_Strings.To_String (Module.Module_Name) & " do" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  # STUNIR generated code" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "end" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-- STUNIR Generated Functional Code" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-- DO-178C Level A" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (algebraic data type)
   overriding procedure Emit_Type
     (Self   : in out Functional_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Language is
         when Haskell =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "data " & Name_Strings.To_String (T.Name) & " = " & Name_Strings.To_String (T.Name) & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  { ");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (T.Fields (I).Name) & " :: String");
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => " }" & ASCII.LF);

         when OCaml =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "type " & Name_Strings.To_String (T.Name) & " = {" & ASCII.LF);
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "  " & Name_Strings.To_String (T.Fields (I).Name) & ": string;");
               Code_Buffers.Append (Source => Output, New_Item => ASCII.LF);
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "}" & ASCII.LF);

         when F_Sharp =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "type " & Name_Strings.To_String (T.Name) & " = {" & ASCII.LF);
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "  " & Name_Strings.To_String (T.Fields (I).Name) & ": string");
               Code_Buffers.Append (Source => Output, New_Item => ASCII.LF);
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "}" & ASCII.LF);

         when Erlang =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "-record(" & Name_Strings.To_String (T.Name) & ", {");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (T.Fields (I).Name));
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "})." & ASCII.LF);

         when Elixir =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  defstruct [");
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (T.Fields (I).Name));
               if I < T.Field_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "]" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "type " & Name_Strings.To_String (T.Name) & " = ..." & ASCII.LF);
      end case;

      Success := True;
   end Emit_Type;

   -- Emit function definition
   overriding procedure Emit_Function
     (Self   : in out Functional_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Language is
         when Haskell =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => Name_Strings.To_String (Func.Name) & " :: ");
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append (Source => Output, New_Item => "a -> ");
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "b" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => Name_Strings.To_String (Func.Name));
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => " " & Name_Strings.To_String (Func.Args (I).Name));
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => " = undefined" & ASCII.LF);

         when OCaml =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  let " & Name_Strings.To_String (Func.Name));
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => " " & Name_Strings.To_String (Func.Args (I).Name));
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => " = ()" & ASCII.LF);

         when F_Sharp =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "let " & Name_Strings.To_String (Func.Name));
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => " " & Name_Strings.To_String (Func.Args (I).Name));
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => " = ()" & ASCII.LF);

         when Erlang =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => Name_Strings.To_String (Func.Name) & "(");
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (Func.Args (I).Name));
               if I < Func.Arg_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ") -> ok." & ASCII.LF);

         when Elixir =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  def " & Name_Strings.To_String (Func.Name) & "(");
            for I in 1 .. Func.Arg_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => Name_Strings.To_String (Func.Args (I).Name));
               if I < Func.Arg_Cnt then
                  Code_Buffers.Append (Source => Output, New_Item => ", ");
               end if;
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ") do" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    :ok" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  end" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "fun " & Name_Strings.To_String (Func.Name) & "() = ..." & ASCII.LF);
      end case;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Functional;
