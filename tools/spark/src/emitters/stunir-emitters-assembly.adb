-- STUNIR Assembly Emitter (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.Assembly is
   pragma SPARK_Mode (On);

   function Get_Target_Name (Target : Assembly_Target) return String is
   begin
      case Target is
         when Target_X86    => return "x86";
         when Target_X86_64 => return "x86_64";
         when Target_ARM    => return "ARM";
         when Target_ARM64  => return "ARM64";
      end case;
   end Get_Target_Name;

   function Get_Syntax_Name (Syntax : Assembly_Syntax) return String is
   begin
      case Syntax is
         when Syntax_Intel => return "Intel";
         when Syntax_ATT   => return "AT&T";
         when Syntax_ARM   => return "ARM";
      end case;
   end Get_Syntax_Name;

   overriding procedure Emit_Module
     (Self    : in out Assembly_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      -- Assembly header
      case Self.Config.Syntax is
         when Syntax_Intel =>
            Append_Line (Gen, "; STUNIR Generated Assembly (Intel Syntax)", Line_Success);
            if not Line_Success then return; end if;
            Append_Line (Gen, ".intel_syntax noprefix", Line_Success);
            if not Line_Success then return; end if;
         when Syntax_ATT =>
            Append_Line (Gen, "# STUNIR Generated Assembly (AT&T Syntax)", Line_Success);
            if not Line_Success then return; end if;
         when Syntax_ARM =>
            Append_Line (Gen, "@ STUNIR Generated Assembly (ARM Syntax)", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Data section
      Append_Line (Gen, ".section .data", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Text section
      Append_Line (Gen, ".section .text", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, ".global _start", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit functions
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Func_Output : IR_Code_Buffer;
            Func_Success : Boolean;
         begin
            Emit_Function (Self, Module.Functions (I), Func_Output, Func_Success);
            if Func_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Func_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type
     (Self    : in out Assembly_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Type_Name : constant String := Get_Type_Name (T);
   begin
      Success := False;
      Initialize (Gen);

      -- Assembly comment for type
      case Self.Config.Syntax is
         when Syntax_Intel | Syntax_ATT =>
            Append_Line (Gen, "; Type: " & Type_Name, Line_Success);
            if not Line_Success then return; end if;
         when Syntax_ARM =>
            Append_Line (Gen, "@ Type: " & Type_Name, Line_Success);
            if not Line_Success then return; end if;
      end case;

      for I in 1 .. T.Field_Cnt loop
         pragma Loop_Invariant (I <= T.Field_Cnt);

         declare
            Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
            Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
         begin
            case Self.Config.Syntax is
               when Syntax_Intel | Syntax_ATT =>
                  Append_Line (Gen, ";   " & Field_Name & ": " & Field_Type, Line_Success);
                  if not Line_Success then return; end if;
               when Syntax_ARM =>
                  Append_Line (Gen, "@   " & Field_Name & ": " & Field_Type, Line_Success);
                  if not Line_Success then return; end if;
            end case;
         end;
      end loop;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self    : in out Assembly_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Func_Name : constant String := Get_Function_Name (Func);
   begin
      Success := False;
      Initialize (Gen);

      -- Function label
      Append_Line (Gen, Func_Name & ":", Line_Success);
      if not Line_Success then return; end if;

      -- Prologue
      Emit_Function_Prologue (Self, Gen, Func, Line_Success);
      if not Line_Success then
         Success := False;
         return;
      end if;

      Increase_Indent (Gen);

      -- Function body
      for I in 1 .. Func.Stmt_Cnt loop
         pragma Loop_Invariant (I <= Func.Stmt_Cnt);

         declare
            Stmt_Data : constant String := Code_Buffers.To_String (Func.Statements (I).Data);
         begin
            if Stmt_Data'Length > 0 then
               Append_Line (Gen, Stmt_Data, Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Decrease_Indent (Gen);

      -- Epilogue
      Emit_Function_Epilogue (Self, Gen, Line_Success);
      if not Line_Success then
         Success := False;
         return;
      end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

   procedure Emit_Function_Prologue
     (Self    : in out Assembly_Emitter;
      Gen     : in out Code_Generator;
      Func    : in     IR_Function;
      Success :    out Boolean)
   is
      Line_Success : Boolean;
   begin
      Success := False;

      Increase_Indent (Gen);

      case Self.Config.Target is
         when Target_X86 | Target_X86_64 =>
            case Self.Config.Syntax is
               when Syntax_Intel =>
                  Append_Line (Gen, "push rbp", Line_Success);
                  if not Line_Success then return; end if;
                  Append_Line (Gen, "mov rbp, rsp", Line_Success);
                  if not Line_Success then return; end if;
               when Syntax_ATT =>
                  Append_Line (Gen, "pushq %rbp", Line_Success);
                  if not Line_Success then return; end if;
                  Append_Line (Gen, "movq %rsp, %rbp", Line_Success);
                  if not Line_Success then return; end if;
               when Syntax_ARM =>
                  null;  -- Not applicable
            end case;
         when Target_ARM | Target_ARM64 =>
            Append_Line (Gen, "push {fp, lr}", Line_Success);
            if not Line_Success then return; end if;
            Append_Line (Gen, "mov fp, sp", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Decrease_Indent (Gen);
      Success := True;
   end Emit_Function_Prologue;

   procedure Emit_Function_Epilogue
     (Self    : in out Assembly_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean)
   is
      Line_Success : Boolean;
   begin
      Success := False;

      Increase_Indent (Gen);

      case Self.Config.Target is
         when Target_X86 | Target_X86_64 =>
            case Self.Config.Syntax is
               when Syntax_Intel =>
                  Append_Line (Gen, "pop rbp", Line_Success);
                  if not Line_Success then return; end if;
                  Append_Line (Gen, "ret", Line_Success);
                  if not Line_Success then return; end if;
               when Syntax_ATT =>
                  Append_Line (Gen, "popq %rbp", Line_Success);
                  if not Line_Success then return; end if;
                  Append_Line (Gen, "ret", Line_Success);
                  if not Line_Success then return; end if;
               when Syntax_ARM =>
                  null;  -- Not applicable
            end case;
         when Target_ARM | Target_ARM64 =>
            Append_Line (Gen, "pop {fp, pc}", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Decrease_Indent (Gen);
      Success := True;
   end Emit_Function_Epilogue;

end STUNIR.Emitters.Assembly;
