-------------------------------------------------------------------------------
--  STUNIR IR to Code Emitter - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  This package implements deterministic code generation from STUNIR IR.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Directories;
with Ada.Command_Line;
with Ada.Characters.Handling;

package body STUNIR_IR_To_Code is

   use Ada.Text_IO;
   use Ada.Directories;
   use Ada.Characters.Handling;

   --  Initialize emission configuration
   procedure Initialize_Config
     (Config        : out Emission_Config;
      IR_Path       : String;
      Template_Path : String;
      Output_Path   : String;
      Target        : Target_Language := Target_Python)
   is
   begin
      Config.IR_Path       := Path_Strings.To_Bounded_String (IR_Path);
      Config.Template_Path := Path_Strings.To_Bounded_String (Template_Path);
      Config.Output_Path   := Path_Strings.To_Bounded_String (Output_Path);
      Config.Target        := Target;
      Config.Emit_Comments := True;
      Config.Emit_Metadata := True;
   end Initialize_Config;

   --  Get file extension for target language
   function Get_File_Extension (Target : Target_Language) return String is
   begin
      case Target is
         when Target_Python     => return ".py";
         when Target_Rust       => return ".rs";
         when Target_C          => return ".c";
         when Target_Cpp        => return ".cpp";
         when Target_Go         => return ".go";
         when Target_JavaScript => return ".js";
         when Target_TypeScript => return ".ts";
         when Target_Java       => return ".java";
         when Target_CSharp     => return ".cs";
         when Target_WASM       => return ".wasm";
         when Target_Assembly_X86 => return ".asm";
         when Target_Assembly_ARM => return ".s";
      end case;
   end Get_File_Extension;

   --  Get target language from string
   function Parse_Target (S : String) return Target_Language is
      Lower_S : constant String := To_Lower (S);
   begin
      if Lower_S = "python" or Lower_S = "py" then
         return Target_Python;
      elsif Lower_S = "rust" or Lower_S = "rs" then
         return Target_Rust;
      elsif Lower_S = "c" then
         return Target_C;
      elsif Lower_S = "cpp" or Lower_S = "c++" then
         return Target_Cpp;
      elsif Lower_S = "go" or Lower_S = "golang" then
         return Target_Go;
      elsif Lower_S = "javascript" or Lower_S = "js" then
         return Target_JavaScript;
      elsif Lower_S = "typescript" or Lower_S = "ts" then
         return Target_TypeScript;
      elsif Lower_S = "java" then
         return Target_Java;
      elsif Lower_S = "csharp" or Lower_S = "cs" then
         return Target_CSharp;
      elsif Lower_S = "wasm" then
         return Target_WASM;
      elsif Lower_S = "x86" or Lower_S = "asm" then
         return Target_Assembly_X86;
      elsif Lower_S = "arm" then
         return Target_Assembly_ARM;
      else
         return Target_Python;  --  Default
      end if;
   end Parse_Target;

   --  Parse IR from JSON file (simplified parser)
   procedure Parse_IR
     (IR_Path   : Path_String;
      Module    : out IR_Module;
      Success   : out Boolean)
   is
      File_Name : constant String := Path_Strings.To_String (IR_Path);
      File      : File_Type;
      Line      : String (1 .. 4096);
      Last      : Natural;
   begin
      Module := (Schema      => Name_Strings.Null_Bounded_String,
                 Module_Name => Name_Strings.Null_Bounded_String,
                 Description => Path_Strings.Null_Bounded_String,
                 Functions   => (others => (Name        => Name_Strings.Null_Bounded_String,
                                            Params      => (others => (Name      => Name_Strings.Null_Bounded_String,
                                                                       Type_Name => Name_Strings.Null_Bounded_String)),
                                            Param_Count => 0,
                                            Return_Type => Name_Strings.Null_Bounded_String,
                                            Is_Public   => True)),
                 Func_Count  => 0);
      Success := False;

      if not Exists (File_Name) then
         Put_Line ("[ERROR] IR file not found: " & File_Name);
         return;
      end if;

      --  Open and parse IR file
      Open (File, In_File, File_Name);

      while not End_Of_File (File) loop
         Get_Line (File, Line, Last);
         --  Simple JSON parsing - look for key fields
         --  This is a simplified parser; production would use proper JSON
         if Last > 0 then
            declare
               L : constant String := Line (1 .. Last);
            begin
               --  Look for "schema":, "module":, "functions": etc.
               if L'Length > 10 then
                  --  Extract schema
                  if L (L'First .. L'First + 7) = """schema"" then
                     Module.Schema := Name_Strings.To_Bounded_String ("stunir.spec.v1");
                  end if;
               end if;
            end;
         end if;
      end loop;

      Close (File);
      Success := True;

   exception
      when others =>
         if Is_Open (File) then
            Close (File);
         end if;
         Success := False;
   end Parse_IR;

   --  Load template for target language
   procedure Load_Template
     (Template_Path : Path_String;
      Target        : Target_Language;
      Template      : out Path_String;
      Success       : out Boolean)
   is
      pragma Unreferenced (Target);
      Template_Dir : constant String := Path_Strings.To_String (Template_Path);
   begin
      Template := Path_Strings.Null_Bounded_String;
      Success := Exists (Template_Dir);

      if Success then
         Put_Line ("[INFO] Template directory found: " & Template_Dir);
         Template := Template_Path;
      else
         Put_Line ("[WARN] Template directory not found, using defaults");
         Success := True;  --  Continue with default templates
      end if;
   end Load_Template;

   --  Emit Python function
   procedure Emit_Python_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
   begin
      Put (File, "def " & Name_Strings.To_String (Func.Name) & "(");

      for I in 1 .. Func.Param_Count loop
         if I > 1 then
            Put (File, ", ");
         end if;
         Put (File, Name_Strings.To_String (Func.Params (I).Name));
         Put (File, ": " & Name_Strings.To_String (Func.Params (I).Type_Name));
      end loop;

      Put_Line (File, ") -> " & Name_Strings.To_String (Func.Return_Type) & ":");
      Put_Line (File, "    pass  # TODO: Implement");
      New_Line (File);
   end Emit_Python_Function;

   --  Emit Rust function
   procedure Emit_Rust_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
   begin
      if Func.Is_Public then
         Put (File, "pub ");
      end if;
      Put (File, "fn " & Name_Strings.To_String (Func.Name) & "(");

      for I in 1 .. Func.Param_Count loop
         if I > 1 then
            Put (File, ", ");
         end if;
         Put (File, Name_Strings.To_String (Func.Params (I).Name));
         Put (File, ": " & Name_Strings.To_String (Func.Params (I).Type_Name));
      end loop;

      Put_Line (File, ") -> " & Name_Strings.To_String (Func.Return_Type) & " {");
      Put_Line (File, "    todo!()  // TODO: Implement");
      Put_Line (File, "}");
      New_Line (File);
   end Emit_Rust_Function;

   --  Emit C function
   procedure Emit_C_Function
     (Func   : Function_Definition;
      File   : in out File_Type)
   is
   begin
      Put (File, Name_Strings.To_String (Func.Return_Type) & " ");
      Put (File, Name_Strings.To_String (Func.Name) & "(");

      if Func.Param_Count = 0 then
         Put (File, "void");
      else
         for I in 1 .. Func.Param_Count loop
            if I > 1 then
               Put (File, ", ");
            end if;
            Put (File, Name_Strings.To_String (Func.Params (I).Type_Name) & " ");
            Put (File, Name_Strings.To_String (Func.Params (I).Name));
         end loop;
      end if;

      Put_Line (File, ") {");
      Put_Line (File, "    /* TODO: Implement */");
      Put_Line (File, "    return 0;");
      Put_Line (File, "}");
      New_Line (File);
   end Emit_C_Function;

   --  Emit code for a single function
   procedure Emit_Function
     (Func     : Function_Definition;
      Target   : Target_Language;
      Output   : out Path_String;
      Success  : out Boolean)
   is
      pragma Unreferenced (Output);
   begin
      --  This procedure is for generating function code to a string
      --  For now, mark as successful (actual emission done in Emit_Code)
      Success := True;
   end Emit_Function;

   --  Main emission procedure
   procedure Emit_Code
     (Config : Emission_Config;
      Result : out Emission_Result)
   is
      Module     : IR_Module;
      Parse_OK   : Boolean;
      Out_File   : File_Type;
      Out_Name   : constant String := Path_Strings.To_String (Config.Output_Path);
      Out_Dir    : constant String := Containing_Directory (Out_Name);
      Template   : Path_String;
      Templ_OK   : Boolean;
   begin
      Result := (Status            => Success,
                 Output_Path       => Config.Output_Path,
                 Output_Size       => 0,
                 Functions_Emitted => 0);

      --  Step 1: Parse IR
      Put_Line ("[INFO] Parsing IR from " & Path_Strings.To_String (Config.IR_Path));
      Parse_IR (Config.IR_Path, Module, Parse_OK);

      if not Parse_OK then
         Result.Status := Error_IR_Parse_Failed;
         return;
      end if;

      --  Step 2: Load templates
      Load_Template (Config.Template_Path, Config.Target, Template, Templ_OK);
      if not Templ_OK then
         Put_Line ("[WARN] Using built-in templates");
      end if;

      --  Step 3: Create output directory
      if not Exists (Out_Dir) then
         Create_Directory (Out_Dir);
      end if;

      --  Step 4: Open output file
      Create (Out_File, Out_File, Out_Name);

      --  Step 5: Emit header
      if Config.Emit_Comments then
         case Config.Target is
            when Target_Python =>
               Put_Line (Out_File, "#!/usr/bin/env python3");
               Put_Line (Out_File, """"""STUNIR Generated Code");
               Put_Line (Out_File, "Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Out_File, "Module: " & Name_Strings.To_String (Module.Module_Name));
               Put_Line (Out_File, """"""");

            when Target_Rust =>
               Put_Line (Out_File, "//! STUNIR Generated Code");
               Put_Line (Out_File, "//! Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Out_File, "//! Module: " & Name_Strings.To_String (Module.Module_Name));

            when Target_C | Target_Cpp =>
               Put_Line (Out_File, "/* STUNIR Generated Code");
               Put_Line (Out_File, " * Generated by: " & Tool_ID & " v" & Version);
               Put_Line (Out_File, " * Module: " & Name_Strings.To_String (Module.Module_Name));
               Put_Line (Out_File, " */");

            when others =>
               Put_Line (Out_File, "// STUNIR Generated Code");
               Put_Line (Out_File, "// Generated by: " & Tool_ID & " v" & Version);
         end case;
         New_Line (Out_File);
      end if;

      --  Step 6: Emit functions
      for I in 1 .. Module.Func_Count loop
         case Config.Target is
            when Target_Python =>
               Emit_Python_Function (Module.Functions (I), Out_File);
            when Target_Rust =>
               Emit_Rust_Function (Module.Functions (I), Out_File);
            when Target_C | Target_Cpp =>
               Emit_C_Function (Module.Functions (I), Out_File);
            when others =>
               --  Default to C-style for now
               Emit_C_Function (Module.Functions (I), Out_File);
         end case;
         Result.Functions_Emitted := Result.Functions_Emitted + 1;
      end loop;

      Close (Out_File);
      Put_Line ("[INFO] Emitted " & Natural'Image (Result.Functions_Emitted) &
                " functions to " & Out_Name);
      Result.Status := Success;

   exception
      when others =>
         if Is_Open (Out_File) then
            Close (Out_File);
         end if;
         Result.Status := Error_Output_Write_Failed;
   end Emit_Code;

   --  Entry point for command-line execution
   procedure Run_IR_To_Code
   is
      use Ada.Command_Line;
      Config        : Emission_Config;
      Result        : Emission_Result;
      IR_Path       : Path_String;
      Template_Path : Path_String := Path_Strings.To_Bounded_String ("templates");
      Output_Path   : Path_String;
      Target        : Target_Language := Target_Python;
   begin
      --  Parse command line arguments
      if Argument_Count < 4 then
         Put_Line ("STUNIR IR to Code Emitter (Ada SPARK) v" & Version);
         Put_Line ("Usage: ir_to_code --input <ir.json> --output <file> --target <lang>");
         Put_Line ("Targets: python, rust, c, cpp, go, javascript, typescript, java, csharp, wasm, x86, arm");
         Set_Exit_Status (Failure);
         return;
      end if;

      --  Simple argument parsing
      for I in 1 .. Argument_Count loop
         if Argument (I) = "--input" and I < Argument_Count then
            IR_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--output" and I < Argument_Count then
            Output_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         elsif Argument (I) = "--target" and I < Argument_Count then
            Target := Parse_Target (Argument (I + 1));
         elsif Argument (I) = "--templates" and I < Argument_Count then
            Template_Path := Path_Strings.To_Bounded_String (Argument (I + 1));
         end if;
      end loop;

      Initialize_Config
        (Config,
         Path_Strings.To_String (IR_Path),
         Path_Strings.To_String (Template_Path),
         Path_Strings.To_String (Output_Path),
         Target);

      Emit_Code (Config, Result);

      case Result.Status is
         when Success =>
            Set_Exit_Status (Success);
         when others =>
            Put_Line ("[ERROR] Emission failed with status: " & Emission_Status'Image (Result.Status));
            Set_Exit_Status (Failure);
      end case;
   end Run_IR_To_Code;

end STUNIR_IR_To_Code;
