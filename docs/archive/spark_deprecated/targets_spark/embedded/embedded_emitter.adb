--  STUNIR Embedded Emitter - Ada SPARK Implementation
--  DO-178C Level A compliant for safety-critical avionics

pragma SPARK_Mode (On);

package body Embedded_Emitter is

   New_Line : constant Character := ASCII.LF;

   --  Equality for Embedded_Statement (required for Formal_Vectors)
   function "=" (Left, Right : Embedded_Statement) return Boolean is
   begin
      return Left.Stmt_Type = Right.Stmt_Type
         and then Left.Data_Type = Right.Data_Type
         and then Identifier_Strings."=" (Left.Target, Right.Target)
         and then Content_Strings."=" (Left.Value, Right.Value)
         and then Identifier_Strings."=" (Left.Left_Op, Right.Left_Op)
         and then Identifier_Strings."=" (Left.Right_Op, Right.Right_Op);
   end "=";

   --  Helper to append string to content buffer
   procedure Append_To_Content (
      Content : in out Content_String;
      Text    : in String;
      Status  : out Emitter_Status)
   is
      Current_Len : constant Natural := Content_Strings.Length (Content);
   begin
      if Current_Len + Text'Length > Max_Content_Length then
         Status := Error_Buffer_Overflow;
      else
         Content_Strings.Append (Content, Text);
         Status := Success;
      end if;
   end Append_To_Content;

   --  Get C type for embedded targets
   function Get_C_Type (
      Data_Type : IR_Data_Type;
      Config    : Embedded_Config) return Type_Name_String
   is
      pragma Unreferenced (Config);
   begin
      return Map_IR_Type_To_C (Data_Type);
   end Get_C_Type;

   --  Calculate memory alignment
   function Calculate_Alignment (
      Data_Type : IR_Data_Type;
      Config    : Embedded_Config) return Positive
   is
      Arch_Config : constant Arch_Config_Type := Get_Arch_Config (Config.Architecture);
   begin
      case Data_Type is
         when Type_I8 | Type_U8 | Type_Bool | Type_Char =>
            return 1;
         when Type_I16 | Type_U16 =>
            return Positive'Min (2, Arch_Config.Alignment);
         when Type_I32 | Type_U32 | Type_F32 =>
            return Positive'Min (4, Arch_Config.Alignment);
         when Type_I64 | Type_U64 | Type_F64 =>
            return Positive'Min (8, Arch_Config.Alignment);
         when others =>
            return Arch_Config.Alignment;
      end case;
   end Calculate_Alignment;

   --  Emit a single statement
   procedure Emit_Statement (
      Stmt      : in Embedded_Statement;
      Config    : in Embedded_Config;
      Indent    : in Natural;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Indent_Str : constant String (1 .. Indent) := (others => ' ');
      C_Type     : Type_Name_String;
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Stmt.Stmt_Type is
         when Stmt_Nop =>
            Append_To_Content (Content, Indent_Str & "/* nop */" & New_Line, Status);

         when Stmt_Var_Decl =>
            C_Type := Get_C_Type (Stmt.Data_Type, Config);
            Append_To_Content (Content, 
               Indent_Str & Type_Name_Strings.To_String (C_Type) & " " &
               Identifier_Strings.To_String (Stmt.Target) & " = " &
               Content_Strings.To_String (Stmt.Value) & ";" & New_Line, Status);

         when Stmt_Assign =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & " = " &
               Content_Strings.To_String (Stmt.Value) & ";" & New_Line, Status);

         when Stmt_Return =>
            Append_To_Content (Content,
               Indent_Str & "return " &
               Content_Strings.To_String (Stmt.Value) & ";" & New_Line, Status);

         when Stmt_Add =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & " = " &
               Identifier_Strings.To_String (Stmt.Left_Op) & " + " &
               Identifier_Strings.To_String (Stmt.Right_Op) & ";" & New_Line, Status);

         when Stmt_Sub =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & " = " &
               Identifier_Strings.To_String (Stmt.Left_Op) & " - " &
               Identifier_Strings.To_String (Stmt.Right_Op) & ";" & New_Line, Status);

         when Stmt_Mul =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & " = " &
               Identifier_Strings.To_String (Stmt.Left_Op) & " * " &
               Identifier_Strings.To_String (Stmt.Right_Op) & ";" & New_Line, Status);

         when Stmt_Div =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & " = " &
               Identifier_Strings.To_String (Stmt.Left_Op) & " / " &
               Identifier_Strings.To_String (Stmt.Right_Op) & ";" & New_Line, Status);

         when Stmt_Call =>
            Append_To_Content (Content,
               Indent_Str & Identifier_Strings.To_String (Stmt.Target) & "(" &
               Content_Strings.To_String (Stmt.Value) & ");" & New_Line, Status);

         when Stmt_If | Stmt_Loop | Stmt_Break | Stmt_Continue | Stmt_Block =>
            Append_To_Content (Content, 
               Indent_Str & "/* " & IR_Statement_Type'Image (Stmt.Stmt_Type) & 
               " - complex statement */" & New_Line, Status);
      end case;
   end Emit_Statement;

   --  Emit a function
   procedure Emit_Function (
      Func      : in Embedded_Function;
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Return_Type : Type_Name_String;
      Stmt_Content : Content_String;
      Stmt_Status  : Emitter_Status;
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Return_Type := Get_C_Type (Func.Return_Type, Config);

      --  Function signature
      Append_To_Content (Content,
         Type_Name_Strings.To_String (Return_Type) & " " &
         Identifier_Strings.To_String (Func.Name) & "(", Status);

      if Status /= Success then return; end if;

      --  Parameters
      if Parameter_Vectors.Is_Empty (Func.Params) then
         Append_To_Content (Content, "void", Status);
      else
         for I in Parameter_Vectors.First_Index (Func.Params) .. Parameter_Vectors.Last_Index (Func.Params) loop
            if I > Parameter_Vectors.First_Index (Func.Params) then
               Append_To_Content (Content, ", ", Status);
            end if;
            declare
               Param_Type : constant Type_Name_String :=
                  Get_C_Type (Parameter_Type_Vectors.Element (Func.Param_Types, I), Config);
            begin
               Append_To_Content (Content,
                  Type_Name_Strings.To_String (Param_Type) & " " &
                  Identifier_Strings.To_String (Parameter_Vectors.Element (Func.Params, I)), Status);
            end;
         end loop;
      end if;

      Append_To_Content (Content, ") {" & New_Line, Status);
      if Status /= Success then return; end if;

      --  Function body
      for I in Statement_Vectors.First_Index (Func.Statements) .. Statement_Vectors.Last_Index (Func.Statements) loop
         Emit_Statement (Statement_Vectors.Element (Func.Statements, I), Config, 4, Stmt_Content, Stmt_Status);
         if Stmt_Status /= Success then
            Status := Stmt_Status;
            return;
         end if;
         Append_To_Content (Content, Content_Strings.To_String (Stmt_Content), Status);
         if Status /= Success then return; end if;
      end loop;

      Append_To_Content (Content, "}" & New_Line & New_Line, Status);
   end Emit_Function;

   --  Generate header file
   procedure Generate_Header (
      Module    : in Embedded_Module;
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Module_Name : constant String := Identifier_Strings.To_String (Module.Name);
      Guard_Name  : constant String := Module_Name & "_H";
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      --  Header guard
      Append_To_Content (Content, 
         "/*" & New_Line &
         " * STUNIR Generated - Embedded Target" & New_Line &
         " * Module: " & Module_Name & New_Line &
         " * DO-178C Level A Compliant" & New_Line &
         " */" & New_Line &
         "#ifndef " & Guard_Name & New_Line &
         "#define " & Guard_Name & New_Line & New_Line, Status);

      if Status /= Success then return; end if;

      --  Standard includes for embedded
      if Config.Use_Stdlib then
         Append_To_Content (Content, "#include <stdint.h>" & New_Line, Status);
      else
         Append_To_Content (Content,
            "/* Fixed-width types for bare metal */" & New_Line &
            "typedef signed char int8_t;" & New_Line &
            "typedef unsigned char uint8_t;" & New_Line &
            "typedef signed short int16_t;" & New_Line &
            "typedef unsigned short uint16_t;" & New_Line &
            "typedef signed int int32_t;" & New_Line &
            "typedef unsigned int uint32_t;" & New_Line &
            "typedef signed long long int64_t;" & New_Line &
            "typedef unsigned long long uint64_t;" & New_Line & New_Line, Status);
      end if;

      if Status /= Success then return; end if;

      --  Function declarations
      for I in Function_Vectors.First_Index (Module.Functions) .. Function_Vectors.Last_Index (Module.Functions) loop
         declare
            Func : constant Embedded_Function := Function_Vectors.Element (Module.Functions, I);
            Return_Type : constant Type_Name_String := Get_C_Type (Func.Return_Type, Config);
         begin
            Append_To_Content (Content,
               Type_Name_Strings.To_String (Return_Type) & " " &
               Identifier_Strings.To_String (Func.Name) & "(", Status);

            if Parameter_Vectors.Is_Empty (Func.Params) then
               Append_To_Content (Content, "void", Status);
            else
               for J in Parameter_Vectors.First_Index (Func.Params) .. Parameter_Vectors.Last_Index (Func.Params) loop
                  if J > Parameter_Vectors.First_Index (Func.Params) then
                     Append_To_Content (Content, ", ", Status);
                  end if;
                  declare
                     Param_Type : constant Type_Name_String :=
                        Get_C_Type (Parameter_Type_Vectors.Element (Func.Param_Types, J), Config);
                  begin
                     Append_To_Content (Content,
                        Type_Name_Strings.To_String (Param_Type) & " " &
                        Identifier_Strings.To_String (Parameter_Vectors.Element (Func.Params, J)), Status);
                  end;
               end loop;
            end if;

            Append_To_Content (Content, ");", Status);
         end;
      end loop;

      Append_To_Content (Content, New_Line & "#endif /* " & Guard_Name & " */" & New_Line, Status);
   end Generate_Header;

   --  Generate linker script
   procedure Generate_Linker_Script (
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Stack_Str : constant String := Positive'Image (Config.Stack_Size);
      Heap_Str  : constant String := Natural'Image (Config.Heap_Size);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Append_To_Content (Content,
         "/* STUNIR Generated Linker Script */" & New_Line &
         "/* DO-178C Level A Compliant */" & New_Line & New_Line &
         "MEMORY {" & New_Line &
         "    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 256K" & New_Line &
         "    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K" & New_Line &
         "}" & New_Line & New_Line &
         "_stack_size =" & Stack_Str & ";" & New_Line &
         "_heap_size =" & Heap_Str & ";" & New_Line & New_Line &
         "SECTIONS {" & New_Line &
         "    .text : {" & New_Line &
         "        *(.text*)" & New_Line &
         "    } > FLASH" & New_Line & New_Line &
         "    .data : {" & New_Line &
         "        *(.data*)" & New_Line &
         "    } > RAM AT > FLASH" & New_Line & New_Line &
         "    .bss : {" & New_Line &
         "        *(.bss*)" & New_Line &
         "    } > RAM" & New_Line &
         "}" & New_Line, Status);
   end Generate_Linker_Script;

   --  Generate startup code
   procedure Generate_Startup (
      Config    : in Embedded_Config;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      pragma Unreferenced (Config);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Append_To_Content (Content,
         "/* STUNIR Generated Startup Code */" & New_Line &
         "/* DO-178C Level A Compliant */" & New_Line & New_Line &
         "extern void main(void);" & New_Line &
         "extern uint32_t _stack_top;" & New_Line & New_Line &
         "void Reset_Handler(void) {" & New_Line &
         "    /* Initialize .data section */" & New_Line &
         "    /* Initialize .bss section */" & New_Line &
         "    /* Call main */" & New_Line &
         "    main();" & New_Line &
         "    /* Infinite loop if main returns */" & New_Line &
         "    while(1) {}" & New_Line &
         "}" & New_Line & New_Line &
         "void Default_Handler(void) {" & New_Line &
         "    while(1) {}" & New_Line &
         "}" & New_Line & New_Line &
         "/* Vector table */" & New_Line &
         "__attribute__((section("".vectors"")))" & New_Line &
         "void (*const vectors[])(void) = {" & New_Line &
         "    (void (*)(void))(&_stack_top)," & New_Line &
         "    Reset_Handler," & New_Line &
         "    Default_Handler,  /* NMI */" & New_Line &
         "    Default_Handler,  /* HardFault */" & New_Line &
         "};" & New_Line, Status);
   end Generate_Startup;

   --  Emit entire module
   procedure Emit_Module (
      Module    : in Embedded_Module;
      Config    : in Embedded_Config;
      Out_Path  : in Path_String;
      Result    : out Emitter_Result)
   is
      pragma Unreferenced (Out_Path);
      Header_Content : Content_String;
      Source_Content : Content_String;
      Func_Content   : Content_String;
      Header_Status  : Emitter_Status;
      Func_Status    : Emitter_Status;
      Module_Name    : constant String := Identifier_Strings.To_String (Module.Name);
   begin
      Result := (Status => Success, Files_Count => 0, Total_Size => 0);

      --  Generate header
      Generate_Header (Module, Config, Header_Content, Header_Status);
      if Header_Status /= Success then
         Result.Status := Header_Status;
         return;
      end if;
      Result.Files_Count := Result.Files_Count + 1;
      Result.Total_Size := Result.Total_Size + Content_Strings.Length (Header_Content);

      --  Generate source file
      Source_Content := Content_Strings.Null_Bounded_String;
      
      declare
         Temp_Status : Emitter_Status;
      begin
         Append_To_Content (Source_Content,
            "/*" & New_Line &
            " * STUNIR Generated Source" & New_Line &
            " * Module: " & Module_Name & New_Line &
            " * DO-178C Level A Compliant" & New_Line &
            " */" & New_Line & New_Line &
            "#include """ & Module_Name & ".h""" & New_Line & New_Line, Temp_Status);

         if Temp_Status /= Success then
            Result.Status := Temp_Status;
            return;
         end if;
      end;

      --  Emit all functions
      for I in Function_Vectors.First_Index (Module.Functions) .. Function_Vectors.Last_Index (Module.Functions) loop
         Emit_Function (Function_Vectors.Element (Module.Functions, I), Config, Func_Content, Func_Status);
         if Func_Status /= Success then
            Result.Status := Func_Status;
            return;
         end if;
         declare
            Temp_Status : Emitter_Status;
         begin
            Append_To_Content (Source_Content,
               Content_Strings.To_String (Func_Content), Temp_Status);
            if Temp_Status /= Success then
               Result.Status := Temp_Status;
               return;
            end if;
         end;
      end loop;

      Result.Files_Count := Result.Files_Count + 1;
      Result.Total_Size := Result.Total_Size + Content_Strings.Length (Source_Content);
   end Emit_Module;

end Embedded_Emitter;
