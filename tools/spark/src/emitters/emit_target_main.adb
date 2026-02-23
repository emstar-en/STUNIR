--  Emit Target Main - Thin CLI wrapper
--  Emits target language code from IR JSON
--  Phase: 3 (Emit)
--  SPARK_Mode: Off (CLI parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with STUNIR_Types;
use STUNIR_Types;
with IR_Parse;
with Emit_Target;

procedure Emit_Target_Main is

   procedure Print_Usage is
   begin
      Ada.Text_IO.Put_Line ("Usage: emit_target -i <input> -o <output> -t <target>");
      Ada.Text_IO.Put_Line ("  -i <input>    Input ir.json file path");
      Ada.Text_IO.Put_Line ("  -o <output>   Output file path (directory will be created)");
      Ada.Text_IO.Put_Line ("  -t <target>   Target language");
      Ada.Text_IO.Put_Line ("  -h            Show this help message");
      Ada.Text_IO.Put_Line ("");
      Ada.Text_IO.Put_Line ("Supported targets:");
      Ada.Text_IO.Put_Line ("  Mainstream: cpp, c, python, rust, go, java, javascript, csharp, swift,");
      Ada.Text_IO.Put_Line ("              kotlin, spark, ada");
      Ada.Text_IO.Put_Line ("  Lisp family: common-lisp, scheme, racket, emacs-lisp, guile, hy, janet,");
      Ada.Text_IO.Put_Line ("               clojure, cljs");
      Ada.Text_IO.Put_Line ("  Prolog family: swi-prolog, gnu-prolog, mercury, prolog (generic)");
      Ada.Text_IO.Put_Line ("  Functional/Formal: futhark, lean4, haskell");
   end Print_Usage;

   function Parse_Target (Target_Str : String) return Target_Language is
   begin
      --  Mainstream languages
      if Target_Str = "cpp" or Target_Str = "c++" then
         return Target_CPP;
      elsif Target_Str = "c" or Target_Str = "c99" then
         return Target_C;
      elsif Target_Str = "python" or Target_Str = "py" then
         return Target_Python;
      elsif Target_Str = "rust" or Target_Str = "rs" then
         return Target_Rust;
      elsif Target_Str = "go" or Target_Str = "golang" then
         return Target_Go;
      elsif Target_Str = "java" then
         return Target_Java;
      elsif Target_Str = "javascript" or Target_Str = "js" then
         return Target_JavaScript;
      elsif Target_Str = "csharp" or Target_Str = "cs" or Target_Str = "c#" then
         return Target_CSharp;
      elsif Target_Str = "swift" then
         return Target_Swift;
      elsif Target_Str = "kotlin" or Target_Str = "kt" then
         return Target_Kotlin;
      elsif Target_Str = "spark" then
         return Target_SPARK;
      elsif Target_Str = "ada" then
         return Target_Ada;
      --  Lisp family
      elsif Target_Str = "common-lisp" or Target_Str = "cl" or Target_Str = "lisp" then
         return Target_Common_Lisp;
      elsif Target_Str = "scheme" then
         return Target_Scheme;
      elsif Target_Str = "racket" then
         return Target_Racket;
      elsif Target_Str = "emacs-lisp" or Target_Str = "elisp" or Target_Str = "el" then
         return Target_Emacs_Lisp;
      elsif Target_Str = "guile" then
         return Target_Guile;
      elsif Target_Str = "hy" then
         return Target_Hy;
      elsif Target_Str = "janet" then
         return Target_Janet;
      elsif Target_Str = "clojure" or Target_Str = "clj" then
         return Target_Clojure;
      elsif Target_Str = "cljs" or Target_Str = "clojurescript" then
         return Target_ClojureScript;
      --  Prolog family
      elsif Target_Str = "swi-prolog" or Target_Str = "swi" then
         return Target_SWI_Prolog;
      elsif Target_Str = "gnu-prolog" or Target_Str = "gprolog" then
         return Target_GNU_Prolog;
      elsif Target_Str = "mercury" then
         return Target_Mercury;
      elsif Target_Str = "prolog" then
         return Target_Prolog;  --  Generic (deprecated)
      --  Functional/Formal
      elsif Target_Str = "futhark" then
         return Target_Futhark;
      elsif Target_Str = "lean4" or Target_Str = "lean" then
         return Target_Lean4;
      elsif Target_Str = "haskell" or Target_Str = "hs" then
         return Target_Haskell;
      else
         return Target_Python;  --  Default
      end if;
   end Parse_Target;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Target      : Target_Language := Target_Python;
   Status      : Status_Code;
   Arg_Index   : Positive := 1;

begin
   --  Parse command-line arguments
   while Arg_Index <= Ada.Command_Line.Argument_Count loop
      declare
         Arg : constant String := Ada.Command_Line.Argument (Arg_Index);
      begin
         if Arg = "-h" or Arg = "--help" then
            Print_Usage;
            Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
            return;
         elsif Arg = "-i" then
            Arg_Index := Arg_Index + 1;
            if Arg_Index <= Ada.Command_Line.Argument_Count then
               Input_Path := Path_Strings.To_Bounded_String (Ada.Command_Line.Argument (Arg_Index));
            else
               Ada.Text_IO.Put_Line ("Error: -i requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         elsif Arg = "-o" then
            Arg_Index := Arg_Index + 1;
            if Arg_Index <= Ada.Command_Line.Argument_Count then
               Output_Path := Path_Strings.To_Bounded_String (Ada.Command_Line.Argument (Arg_Index));
            else
               Ada.Text_IO.Put_Line ("Error: -o requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         elsif Arg = "-t" then
            Arg_Index := Arg_Index + 1;
            if Arg_Index <= Ada.Command_Line.Argument_Count then
               Target := Parse_Target (Ada.Command_Line.Argument (Arg_Index));
            else
               Ada.Text_IO.Put_Line ("Error: -t requires an argument");
               Print_Usage;
               Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
               return;
            end if;
         else
            Ada.Text_IO.Put_Line ("Error: Unknown option: " & Arg);
            Print_Usage;
            Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
            return;
         end if;
         Arg_Index := Arg_Index + 1;
      end;
   end loop;

   --  Validate required arguments
   if Path_Strings.Length (Input_Path) = 0 then
      Ada.Text_IO.Put_Line ("Error: Input path required (-i)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   if Path_Strings.Length (Output_Path) = 0 then
      Ada.Text_IO.Put_Line ("Error: Output path required (-o)");
      Print_Usage;
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
      return;
   end if;

   --  Process
   declare
      IR   : IR_Data;
      Code : Code_String;
   begin
      Ada.Text_IO.Put_Line ("Parsing IR: " & Path_Strings.To_String (Input_Path));
      IR_Parse.Parse_IR_File (Input_Path, IR, Status);
      
      if Status /= Success then
         Ada.Text_IO.Put_Line ("Error: Parse failed with status " & Status_Code'Image (Status));
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
         return;
      end if;

      Ada.Text_IO.Put_Line ("Emitting target: " & Target_Language'Image (Target));
      Emit_Target.Emit_Single_Target (IR, Target, Code, Status);
      
      if Status /= Success then
         Ada.Text_IO.Put_Line ("Error: Emit failed with status " & Status_Code'Image (Status));
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
         return;
      end if;

      --  Ensure output directory exists
      declare
         Out_Dir : constant String := Ada.Directories.Containing_Directory (Path_Strings.To_String (Output_Path));
      begin
         if Out_Dir'Length > 0 and then not Ada.Directories.Exists (Out_Dir) then
            Ada.Directories.Create_Path (Out_Dir);
         end if;
      exception
         when others => null;  --  Directory might be current dir
      end;

      --  Write output
      declare
         Output_File : Ada.Text_IO.File_Type;
      begin
         Ada.Text_IO.Create (Output_File, Ada.Text_IO.Out_File, Path_Strings.To_String (Output_Path));
         Ada.Text_IO.Put (Output_File, Code_Strings.To_String (Code));
         Ada.Text_IO.Close (Output_File);
      exception
         when others =>
            Ada.Text_IO.Put_Line ("Error: Failed to write output file");
            Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
            return;
      end;

      Ada.Text_IO.Put_Line ("Output written to: " & Path_Strings.To_String (Output_Path));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   end;

end Emit_Target_Main;
