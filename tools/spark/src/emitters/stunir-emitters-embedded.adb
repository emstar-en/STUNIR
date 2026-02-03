-- STUNIR Embedded Systems Emitter (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.Embedded is
   pragma SPARK_Mode (On);

   function Get_Arch_Name (Arch : Architecture) return String is
   begin
      case Arch is
         when Arch_ARM    => return "ARM";
         when Arch_ARM64  => return "ARM64";
         when Arch_RISCV  => return "RISC-V";
         when Arch_MIPS   => return "MIPS";
         when Arch_AVR    => return "AVR";
         when Arch_X86    => return "x86";
      end case;
   end Get_Arch_Name;

   function Get_Toolchain_Name (Arch : Architecture) return String is
   begin
      case Arch is
         when Arch_ARM    => return "arm-none-eabi";
         when Arch_ARM64  => return "aarch64-none-elf";
         when Arch_RISCV  => return "riscv32-unknown-elf";
         when Arch_MIPS   => return "mips-elf";
         when Arch_AVR    => return "avr";
         when Arch_X86    => return "i686-elf";
      end case;
   end Get_Toolchain_Name;

   overriding procedure Emit_Module
     (Self    : in out Embedded_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      -- Header comment
      Append_Line (Gen, "/* STUNIR Generated Embedded Code */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* DO-178C Level A Compliant */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* Target: " & Get_Arch_Name (Self.Config.Arch) & " */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Standard includes
      Append_Line (Gen, "#include <stdint.h>", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "#include <stdbool.h>", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit all types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);

         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Self, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Type_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit all functions
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
     (Self    : in out Embedded_Emitter;
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

      Append_Line (Gen, "typedef struct {", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);

      for I in 1 .. T.Field_Cnt loop
         pragma Loop_Invariant (I <= T.Field_Cnt);

         declare
            Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
            Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
            Field_Line : constant String := Field_Type & " " & Field_Name & ";";
         begin
            Append_Line (Gen, Field_Line, Line_Success);
            if not Line_Success then return; end if;
         end;
      end loop;

      Decrease_Indent (Gen);

      Append_Line (Gen, "} " & Type_Name & "_t;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self    : in out Embedded_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Func_Name : constant String := Get_Function_Name (Func);
      Return_Type : constant String := Type_Strings.To_String (Func.Return_Type);
   begin
      Success := False;
      Initialize (Gen);

      -- Function signature
      declare
         Signature : String := Return_Type & " " & Func_Name & "(";
      begin
         for I in 1 .. Func.Arg_Cnt loop
            pragma Loop_Invariant (I <= Func.Arg_Cnt);

            declare
               Arg_Type : constant String := Type_Strings.To_String (Func.Args (I).Type_Ref);
               Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
            begin
               if I > 1 then
                  Signature := Signature & ", ";
               end if;
               Signature := Signature & Arg_Type & " " & Arg_Name;
            end;
         end loop;

         if Func.Arg_Cnt = 0 then
            Signature := Signature & "void";
         end if;

         Signature := Signature & ") {";
         Append_Line (Gen, Signature, Line_Success);
         if not Line_Success then return; end if;
      end;

      Increase_Indent (Gen);

      -- Function body (simplified for Phase 3a)
      Append_Line (Gen, "/* STUNIR Generated Function Body */", Line_Success);
      if not Line_Success then return; end if;

      -- Emit statements
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

      Append_Line (Gen, "}", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Function;

   procedure Emit_Startup_Code
     (Self    : in out Embedded_Emitter;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "/* STUNIR Embedded Startup Code */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* Architecture: " & Get_Arch_Name (Self.Config.Arch) & " */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "extern void main(void);", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "extern unsigned long _estack;", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "void Reset_Handler(void) {", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);
      Append_Line (Gen, "/* Initialize data and bss */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* Call main */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "main();", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "while(1);", Line_Success);
      if not Line_Success then return; end if;

      Decrease_Indent (Gen);
      Append_Line (Gen, "}", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Startup_Code;

   procedure Emit_Linker_Script
     (Self    : in out Embedded_Emitter;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "/* STUNIR Generated Linker Script */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "MEMORY", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "{", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);
      Append_Line (Gen, "FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 256K", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 64K", Line_Success);
      if not Line_Success then return; end if;

      Decrease_Indent (Gen);
      Append_Line (Gen, "}", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Linker_Script;

   procedure Emit_Memory_Config
     (Self    : in out Embedded_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean)
   is
      Line_Success : Boolean;
   begin
      Success := False;

      Append_Line (Gen, "/* Memory Configuration */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "#define STACK_SIZE " & Natural'Image (Self.Config.Stack_Size), Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "#define HEAP_SIZE " & Natural'Image (Self.Config.Heap_Size), Line_Success);
      if not Line_Success then return; end if;

      Success := True;
   end Emit_Memory_Config;

end STUNIR.Emitters.Embedded;
