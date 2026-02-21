-- STUNIR Lisp Family Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3b: Language Family Emitters


with Semantic_IR.Types; use Semantic_IR.Types;
with STUNIR.Emitters.AST_Render;
with STUNIR.Emitters.CodeGen; use STUNIR.Emitters.CodeGen;

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
         when Common_Lisp | Clojure | ClojureScript | Racket =>
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
         when Clojure | ClojureScript =>
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
     (Prim_Type : Semantic_IR.Types.IR_Primitive_Type;
      Dialect   : Lisp_Dialect) return String
   is
      pragma Unreferenced (Dialect);
   begin
      case Prim_Type is
         when Semantic_IR.Types.Type_String =>
            return "string";
         when Semantic_IR.Types.Type_I8 | Semantic_IR.Types.Type_I16 | Semantic_IR.Types.Type_I32 | Semantic_IR.Types.Type_I64 =>
            return "integer";
         when Semantic_IR.Types.Type_U8 | Semantic_IR.Types.Type_U16 | Semantic_IR.Types.Type_U32 | Semantic_IR.Types.Type_U64 =>
            return "integer";
         when Semantic_IR.Types.Type_F32 | Semantic_IR.Types.Type_F64 =>
            return "float";
         when Semantic_IR.Types.Type_Bool =>
            return "boolean";
         when Semantic_IR.Types.Type_Void =>
            return "nil";
         when Semantic_IR.Types.Type_Char =>
            return "char";
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
      Module : in     Semantic_IR.Modules.IR_Module;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Comment_Prefix : constant String := Get_Comment_Prefix (Self.Config.Dialect);
      Module_Name : constant String := Semantic_IR.Types.Name_Strings.To_String (Module.Module_Name);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      -- Header comment
      Emit_Atom (Output, Comment_Prefix & "STUNIR Generated " & Lisp_Dialect'Image (Self.Config.Dialect) & " Code", Temp_Success);
      if not Temp_Success then return; end if;
      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Module/Package/Namespace declaration
      case Self.Config.Dialect is
         when Common_Lisp | Clojure | ClojureScript =>
            Emit_List_Start (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Get_Module_Syntax (Self.Config.Dialect), Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Space (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Atom (Output, Module_Name, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;
         when others =>
            null;
      end case;

      Emit_Newline (Output, Temp_Success);
      if not Temp_Success then return; end if;

      -- Emit functions from declarations
      for I in 1 .. Module.Decl_Count loop
         declare
            Decl_Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
              STUNIR.Emitters.Node_Table.Lookup (Nodes, Module.Declarations (I));
         begin
            if Decl_Index > 0 then
               declare
                  Decl : constant STUNIR.Emitters.Node_Table.Declaration_Record :=
                    STUNIR.Emitters.Node_Table.Get_Declaration (Nodes, Decl_Index);
               begin
                  if Decl.Kind = Semantic_IR.Types.Kind_Function_Decl then
                     declare
                        Func_Output : IR_Code_Buffer;
                        Func_Success : Boolean;
                        F : constant Semantic_IR.Declarations.Function_Declaration := Decl.Func;
                     begin
                        Emit_Function (Self, F, Nodes, Func_Output, Func_Success);
                        if Func_Success then
                           Code_Buffers.Append (Output, Code_Buffers.To_String (Func_Output));
                           Emit_Newline (Output, Temp_Success);
                           if not Temp_Success then return; end if;
                        end if;
                     end;
                  end if;
               end;
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
      T      : in     Semantic_IR.Declarations.Type_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      pragma Unreferenced (Self, Nodes, T);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := True;
   end Emit_Type;

   ----------------------------------------------------------------------------
   -- Emit_Function
   ----------------------------------------------------------------------------
   procedure Emit_Function
     (Self   : in out Lisp_Emitter;
      Func   : in     Semantic_IR.Declarations.Function_Declaration;
      Nodes  : in     STUNIR.Emitters.Node_Table.Node_Table;
      Output :    out STUNIR.Emitters.CodeGen.IR_Code_Buffer;
      Success:    out Boolean)
   is
      Temp_Success : Boolean;
      Func_Name : constant String := Semantic_IR.Types.Name_Strings.To_String (Func.Base.Decl_Name);
   begin
      Output := Code_Buffers.Null_Bounded_String;
      Success := False;

      case Self.Config.Dialect is
         when Common_Lisp | Scheme | Clojure | ClojureScript =>
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
            Emit_Atom (Output, "[]", Temp_Success);
            if not Temp_Success then return; end if;
            Emit_Newline (Output, Temp_Success);
            if not Temp_Success then return; end if;

            if Semantic_IR.Nodes.Is_Valid_Node_ID (Func.Body_ID) then
               declare
                  Stmt_Out : IR_Code_Buffer;
                  Stmt_Ok  : Boolean;
               begin
                  STUNIR.Emitters.AST_Render.Render_Statement (Nodes, Func.Body_ID, Stmt_Out, Stmt_Ok);
                  if Stmt_Ok then
                     Emit_Atom (Output, "  " & Code_Buffers.To_String (Stmt_Out), Temp_Success);
                  else
                     Emit_Atom (Output, "  nil", Temp_Success);
                  end if;
               end;
            else
               Emit_Atom (Output, "  nil", Temp_Success);
            end if;

            Emit_List_End (Output, Temp_Success);
            if not Temp_Success then return; end if;
         when others =>
            Emit_Atom (Output, Get_Comment_Prefix (Self.Config.Dialect) & "Function: " & Func_Name, Temp_Success);
      end case;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Lisp;
