-- STUNIR Lisp Family Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters

package body STUNIR.Emitters.Lisp is
   pragma SPARK_Mode (On);

   New_Line : constant Character := Character'Val (10);
   Space    : constant Character := ' ';

   ----------------------------------------------------------------------------
   -- Get_Comment_Prefix
   ----------------------------------------------------------------------------
   function Get_Comment_Prefix (Dialect : Lisp_Dialect) return String is
   begin
      case Dialect is
         when Common_Lisp | Clojure | Racket =>
            return ";;; ";
         when Scheme | Emacs_Lisp | Guile =>
            return ";; ";
         when Hy =>
            return ";; ";
         when Janet =>
            return "# ";
      end case;
   end Get_Comment_Prefix;

   ----------------------------------------------------------------------------
   -- Get_Module_Syntax
   ----------------------------------------------------------------------------
   function Get_Module_Syntax (Dialect : Lisp_Dialect) return String is
   begin
      case Dialect is
         when Common_Lisp =>
            return "defpackage";
         when Scheme | Guile =>
            return "define-library";
         when Clojure =>
            return "ns";
         when Racket =>
            return "#lang racket";
         when Emacs_Lisp =>
            return "provide";
         when Hy =>
            return "import";
         when Janet =>
            return "import";
      end case;
   end Get_Module_Syntax;

   ----------------------------------------------------------------------------
   -- Map_Type_To_Lisp
   ----------------------------------------------------------------------------
   function Map_Type_To_Lisp
     (Prim_Type : IR_Primitive_Type;
      Dialect   : Lisp_Dialect) return String
   is
      pragma Unreferenced (Dialect);
   begin
      case Prim_Type is
         when Type_String =>
            return "string";
         when Type_Int | Type_I8 | Type_I16 | Type_I32 | Type_I64 =>
            return "integer";
         when Type_U8 | Type_U16 | Type_U32 | Type_U64 =>
            return "integer";
         when Type_Float | Type_F32 | Type_F64 =>
            return "float";
         when Type_Bool =>
            return "boolean";
         when Type_Void =>
            return "nil";
      end case;
   end Map_Type_To_Lisp;

   ----------------------------------------------------------------------------
   -- S-Expression Utilities
   ----------------------------------------------------------------------------
   procedure Emit_List_Start
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) < Max_Code_Length then
         Code_Buffers.Append (Buffer, "(");
         Success := True;
      end if;
   end Emit_List_Start;

   procedure Emit_List_End
     (Buffer  : in out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) < Max_Code_Length then
         Code_Buffers.Append (Buffer, ")");
         Success := True;
      end if;
   end Emit_List_End;

   procedure Emit_Atom
     (Buffer  : in out IR_Code_Buffer;
      Atom    : in     String;
      Success :    out Boolean)
   is
   begin
      Success := False;
      if Code_Buffers.Length (Buffer) + Atom'Length <= Max_Code_Length then
         Code_Buffers.Append (Buffer, Atom);
         Success := True;
      end if;
   end Emit_Atom;

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

   ----------------------------------------------------------------------------
   -- Emit_Module
   ----------------------------------------------------------------------------
   procedure Emit_Module
     (Self   : in out Lisp_Emitter;
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
      Emit_Atom (Output, Comment_Prefix & "STUNIR Generated " &
        Lisp_Dialect'Image (Self.Config.Dialect) & " Code", Temp_Success);
      if not Temp_Success then return; end if;
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      Emit_Atom (Output, Comment_Prefix & "DO-178C Level A Compliant", Temp_Success);
      if not Temp_Success then return; end if;
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Module/Package/Namespace declaration
      case Self.Config.Dialect is
         when Common_Lisp =>
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defpackage", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, ":" & Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "in-package", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, ":" & Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Scheme =>
            if Self.Config.Scheme_Std = R7RS then
               Emit_List_Start (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, "define-library", Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Space (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_List_Start (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, Module_Name, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_List_End (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_List_End (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

         when Clojure =>
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "ns", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Racket =>
            Emit_Atom (Output, "#lang racket/base", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Emacs_Lisp =>
            Emit_Atom (Output, ";;; " & Module_Name & ".el --- ", Temp_Success);
            if not Temp_Success then return; end if;
            if Module_Doc'Length > 0 then
               Emit_Atom (Output, Module_Doc, Temp_Success);
               if not Temp_Success then return; end if;
            end if;
            Emit_Atom (Output, "  -*- lexical-binding: t; -*-", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Guile =>
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "define-module", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Hy =>
            Emit_Atom (Output, ";; Hy Module", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Janet =>
            Emit_Atom (Output, "# Janet Module: " & Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
      end case;

      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Emit types
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

      -- Emit functions
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
     (Self   : in out Lisp_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Type_Name : constant String := Name_Strings.To_String (T.Name);
      Type_Doc  : constant String := Doc_Strings.To_String (T.Docstring);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      case Self.Config.Dialect is
         when Common_Lisp =>
            -- (defstruct type-name ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defstruct", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Type_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Clojure =>
            -- (defrecord TypeName [field1 field2])
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defrecord", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Type_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. T.Field_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (T.Fields (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < T.Field_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when others =>
            -- Generic struct comment
            Emit_Atom (Output, Get_Comment_Prefix (Self.Config.Dialect) & 
              "Type: " & Type_Name, Temp_Success);
            if not Temp_Success then return; end if;
      end case;

      Success := True;
   end Emit_Type;

   ----------------------------------------------------------------------------
   -- Emit_Function
   ----------------------------------------------------------------------------
   procedure Emit_Function
     (Self   : in out Lisp_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Func_Name : constant String := Name_Strings.To_String (Func.Name);
      Func_Doc  : constant String := Doc_Strings.To_String (Func.Docstring);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      case Self.Config.Dialect is
         when Common_Lisp =>
            -- (defun function-name (args) ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defun", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            -- Argument list
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < Func.Arg_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

            -- Docstring
            if Self.Config.Include_Docs and Func_Doc'Length > 0 then
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, "  """ & Func_Doc & """", Temp_Success);
               if not Temp_Success then return; end if;
            end if;

            -- Body
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  nil", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Scheme =>
            -- (define (function-name args) ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "define", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Space (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  #f", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Clojure =>
            -- (defn function-name [args] ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defn", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;

            -- Docstring
            if Self.Config.Include_Docs and Func_Doc'Length > 0 then
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, "  """ & Func_Doc & """", Temp_Success);
               if not Temp_Success then return; end if;
            end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  [", Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < Func.Arg_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_Atom (Output, "]", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  nil", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Racket =>
            -- (define (function-name args) ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "define", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Space (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  #f", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Emacs_Lisp =>
            -- (defun function-name (args) ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defun", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < Func.Arg_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Include_Docs and Func_Doc'Length > 0 then
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, "  """ & Func_Doc & """", Temp_Success);
               if not Temp_Success then return; end if;
            end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  nil", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Guile =>
            -- (define (function-name args) ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "define", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Space (Output, Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
            end loop;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  #f", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Hy =>
            -- (defn function-name [args] ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defn", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Atom (Output, "[", Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < Func.Arg_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_Atom (Output, "]", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  None", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;

         when Janet =>
            -- (defn function-name [args] ...)
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "defn", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Func_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Self.Config.Include_Docs and Func_Doc'Length > 0 then
               Emit_Atom (Output, "  """ & Func_Doc & """", Temp_Success);
               if not Temp_Success then return; end if;
               Emit_Newline (Output, Temp_Success);
               if not Temp_Success then return; end if;
            end if;

            Emit_Atom (Output, "  [", Temp_Success);
            if not Temp_Success then return; end if;
            for I in 1 .. Func.Arg_Cnt loop
               Emit_Atom (Output, Name_Strings.To_String (Func.Args (I).Name), Temp_Success);
               if not Temp_Success then return; end if;
               if I < Func.Arg_Cnt then
                  Emit_Space (Output, Temp_Success);
                  if not Temp_Success then return; end if;
               end if;
            end loop;
            Emit_Atom (Output, "]", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, "  nil", Temp_Success);
            if not Temp_Success then return; end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
      end case;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Lisp;
