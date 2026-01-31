-- STUNIR Prolog Family Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

package body STUNIR.Emitters.Prolog is
   pragma SPARK_Mode (On);

   New_Line : constant Character := Character'Val (10);
   Space    : constant Character := ' ';

   ----------------------------------------------------------------------------
   -- Get_Comment_Prefix
   ----------------------------------------------------------------------------
   function Get_Comment_Prefix (Dialect : Prolog_Dialect) return String is
   begin
      case Dialect is
         when SWI_Prolog | SICStus | YAP | XSB | ECLiPSe =>
            return "%% ";
         when GNU_Prolog | Ciao | BProlog =>
            return "% ";
      end case;
   end Get_Comment_Prefix;

   ----------------------------------------------------------------------------
   -- Get_Module_Syntax
   ----------------------------------------------------------------------------
   function Get_Module_Syntax (Dialect : Prolog_Dialect) return String is
   begin
      case Dialect is
         when SWI_Prolog | SICStus | YAP | XSB | ECLiPSe | Ciao =>
            return "module";
         when GNU_Prolog =>
            return "% module";
         when BProlog =>
            return "module";
      end case;
   end Get_Module_Syntax;

   ----------------------------------------------------------------------------
   -- Map_Type_To_Prolog
   ----------------------------------------------------------------------------
   function Map_Type_To_Prolog
     (Prim_Type : IR_Primitive_Type;
      Dialect   : Prolog_Dialect) return String
   is
      pragma Unreferenced (Dialect);
   begin
      case Prim_Type is
         when Type_String =>
            return "atom";
         when Type_Int | Type_I8 | Type_I16 | Type_I32 | Type_I64 =>
            return "integer";
         when Type_U8 | Type_U16 | Type_U32 | Type_U64 =>
            return "integer";
         when Type_Float | Type_F32 | Type_F64 =>
            return "float";
         when Type_Bool =>
            return "boolean";
         when Type_Void =>
            return "true";
      end case;
   end Map_Type_To_Prolog;

   ----------------------------------------------------------------------------
   -- Dialect Feature Support
   ----------------------------------------------------------------------------
   function Supports_Tabling (Dialect : Prolog_Dialect) return Boolean is
   begin
      return Dialect in XSB | YAP;
   end Supports_Tabling;

   function Supports_CLP (Dialect : Prolog_Dialect) return Boolean is
   begin
      return Dialect in SWI_Prolog | GNU_Prolog | SICStus | ECLiPSe | BProlog;
   end Supports_CLP;

   function Supports_Assertions (Dialect : Prolog_Dialect) return Boolean is
   begin
      return Dialect = Ciao;
   end Supports_Assertions;

   ----------------------------------------------------------------------------
   -- Prolog Utilities
   ----------------------------------------------------------------------------
   procedure Emit_Space
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) < Max_Code_Length then
         Code_Buffers.Append (Buffer, "" & Space);
         Success := True;
      end if;
   end Emit_Space;

   procedure Emit_Newline
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) < Max_Code_Length then
         Code_Buffers.Append (Buffer, "" & New_Line);
         Success := True;
      end if;
   end Emit_Newline;

   procedure Emit_Predicate_Head
     (Buffer         : in out IR_Code_Buffer;
      Predicate_Name : in     String;
      Arity          : in     Natural;
      Success        :    out Boolean)
   is
      Temp_Success : Boolean;
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) + Predicate_Name'Length + 10 <= Max_Code_Length then
         Code_Buffers.Append (Buffer, Predicate_Name);
         if Arity > 0 then
            Code_Buffers.Append (Buffer, "(");
            for I in 1 .. Arity loop
               Code_Buffers.Append (Buffer, "_");
               if I < Arity then
                  Code_Buffers.Append (Buffer, ", ");
               end if;
            end loop;
            Code_Buffers.Append (Buffer, ")");
         end if;
         Success := True;
      end if;
   end Emit_Predicate_Head;

   procedure Emit_Clause_Separator
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) + 4 <= Max_Code_Length then
         Code_Buffers.Append (Buffer, " :- ");
         Success := True;
      end if;
   end Emit_Clause_Separator;

   procedure Emit_Body_Goal
     (Buffer  : in out IR_Code_Buffer;
      Goal    : in     String;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) + Goal'Length <= Max_Code_Length then
         Code_Buffers.Append (Buffer, Goal);
         Success := True;
      end if;
   end Emit_Body_Goal;

   ----------------------------------------------------------------------------
   -- Emit_Module
   ----------------------------------------------------------------------------
   procedure Emit_Module
     (Self   : in out Prolog_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Comment_Prefix : constant String := Get_Comment_Prefix (Self.Config.Dialect);
      Module_Name : constant String := Name_Strings.To_String (Module.Module_Name);
      Module_Doc  : constant String := Doc_Strings.To_String (Module.Docstring);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      -- Header comment
      Code_Buffers.Append (Output, Comment_Prefix & "STUNIR Generated " &
        Prolog_Dialect'Image (Self.Config.Dialect) & " Code");
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      Code_Buffers.Append (Output, Comment_Prefix & "DO-178C Level A Compliant");
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Module declaration
      case Self.Config.Dialect is
         when SWI_Prolog =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ", []).");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Use_CLP then
               Code_Buffers.Append (Output, ":- use_module(library(clpfd)).");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

         when GNU_Prolog =>
            Code_Buffers.Append (Output, "% Module: " & Module_Name);
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Use_CLP then
               Code_Buffers.Append (Output, ":- include('clpfd.pl').");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

         when SICStus =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ", []).");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Use_CLP then
               Code_Buffers.Append (Output, ":- use_module(library(clpfd)).");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

         when YAP =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ", []).");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Use_CLP then
               Code_Buffers.Append (Output, ":- use_module(library(clpfd)).");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

         when XSB =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ", []).");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Ciao =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ", [], [assertions]).");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when BProlog =>
            Code_Buffers.Append (Output, "% Module: " & Module_Name);
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when ECLiPSe =>
            Code_Buffers.Append (Output, ":- module(" & Module_Name & ").");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Use_CLP then
               Code_Buffers.Append (Output, ":- lib(ic).");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;
      end case;

      -- Emit types (as comments for most dialects)
      for I in 1 .. Module.Type_Cnt loop
         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Self, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Code_Buffers.Append (Output, Code_Buffers.To_String (Type_Output));
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit functions (as predicates)
      for I in 1 .. Module.Func_Cnt loop
         declare
            Func_Output : IR_Code_Buffer;
            Func_Success : Boolean;
         begin
            Emit_Function (Self, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Code_Buffers.Append (Output, Code_Buffers.To_String (Func_Output));
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;
         end;
      end loop;

      Success := True;
   end Emit_Module;

   ----------------------------------------------------------------------------
   -- Emit_Type
   ----------------------------------------------------------------------------
   procedure Emit_Type
     (Self   : in out Prolog_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Type_Name : constant String := Name_Strings.To_String (T.Name);
      Type_Doc  : constant String := Doc_Strings.To_String (T.Docstring);
      Comment_Prefix : constant String := Get_Comment_Prefix (Self.Config.Dialect);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      -- Most Prolog dialects don't have struct types, so emit as comments
      case Self.Config.Dialect is
         when Ciao =>
            -- Ciao has regtype for defining types
            Code_Buffers.Append (Output, ":- regtype " & Type_Name & "/1.");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Code_Buffers.Append (Output, Type_Name & "(" & Type_Name & ").");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when others =>
            -- Generic type comment
            Code_Buffers.Append (Output, Comment_Prefix & "Type: " & Type_Name);
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Type_Doc'Length > 0 then
               Code_Buffers.Append (Output, Comment_Prefix & Type_Doc);
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

            Code_Buffers.Append (Output, Comment_Prefix & "  Fields:");
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append (Output, Comment_Prefix & "    " &
                 Name_Strings.To_String (T.Fields (I).Name) & ": " &
                 Type_Strings.To_String (T.Fields (I).Type_Ref));
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end loop;
      end case;

      Success := True;
   end Emit_Type;

   ----------------------------------------------------------------------------
   -- Emit_Function
   ----------------------------------------------------------------------------
   procedure Emit_Function
     (Self   : in out Prolog_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Func_Name : constant String := Name_Strings.To_String (Func.Name);
      Func_Doc  : constant String := Doc_Strings.To_String (Func.Docstring);
      Comment_Prefix : constant String := Get_Comment_Prefix (Self.Config.Dialect);
      Arity : constant Natural := Func.Arg_Cnt + 1;  -- +1 for result argument
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      -- Function docstring as comment
      if Func_Doc'Length > 0 then
         Code_Buffers.Append (Output, Comment_Prefix & Func_Doc);
         Emit_Newline (Output, Temp_Success);
         if not Temp_Success then return; end if;
      end if;

      -- Tabling annotation (if supported and enabled)
      if Self.Config.Use_Tabling and Supports_Tabling (Self.Config.Dialect) then
         case Self.Config.Dialect is
            when XSB =>
               Code_Buffers.Append (Output, ":- table " & Func_Name & "/" &
                 Natural'Image (Arity)(2 .. Natural'Image (Arity)'Last) & ".");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;

            when YAP =>
               Code_Buffers.Append (Output, ":- table " & Func_Name & "/" &
                 Natural'Image (Arity)(2 .. Natural'Image (Arity)'Last) & ".");
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;

            when others =>
               null;
         end case;
      end if;

      -- Assertion annotation (Ciao only)
      if Self.Config.Use_Assertions and Self.Config.Dialect = Ciao then
         Code_Buffers.Append (Output, ":- pred " & Func_Name & "(");
         for I in 1 .. Func.Arg_Cnt loop
            Code_Buffers.Append (Output, "+");
            if I < Func.Arg_Cnt then
               Code_Buffers.Append (Output, ",");
            end if;
         end loop;
         Code_Buffers.Append (Output, ",-).");
         Emit_Newline (Output, Temp_Success);
         if not Temp_Success then return; end if;
      end if;

      -- Predicate head (functional to logic conversion)
      -- function f(x, y) -> z becomes predicate f(X, Y, Z)
      Code_Buffers.Append (Output, Func_Name & "(");

      -- Input arguments
      for I in 1 .. Func.Arg_Cnt loop
         declare
            Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
            Capitalized : String := Arg_Name;
         begin
            -- Capitalize first letter for Prolog variable convention
            if Capitalized'Length > 0 and then Capitalized (Capitalized'First) in 'a' .. 'z' then
               Capitalized (Capitalized'First) :=
                 Character'Val (Character'Pos (Capitalized (Capitalized'First)) - 32);
            end if;
            Code_Buffers.Append (Output, Capitalized);
            Code_Buffers.Append (Output, ", ");
         end;
      end loop;

      -- Output argument (result)
      Code_Buffers.Append (Output, "Result)");

      -- Clause body
      Emit_Clause_Separator (Output, Temp_Success);
      if not Temp_Success then return; end if;

      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Simple body (placeholder - real implementation would translate IR statements)
      Code_Buffers.Append (Output, "    true.");
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Prolog;
