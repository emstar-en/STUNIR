--  Code Emitter Main Program
--  Command-line entry point for ir.json to target code conversion
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);  --  Command-line parsing not in SPARK

with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
use STUNIR_Types;
with Emit_Target;

procedure Code_Emitter_Main is
   package ACL renames Ada.Command_Line;
   use Ada.Text_IO;

   procedure Print_Usage is
   begin
      Put_Line ("Usage: code_emitter <input_ir.json> <target> <output_file>");
      Put_Line ("  input_ir.json    Input ir.json file path");
      Put_Line ("  target           Target language (python, cpp, c, rust, etc.)");
      Put_Line ("  output_file      Output file path");
      Put_Line ("  -h               Show this help message");
      Put_Line ("");
      Put_Line ("Supported targets:");
      Put_Line ("  Mainstream: cpp, c, python, rust, go, java, javascript, csharp, swift,");
      Put_Line ("              kotlin, spark, ada");
      Put_Line ("  Lisp family: common-lisp, scheme, racket, emacs-lisp, guile, hy, janet,");
      Put_Line ("               clojure, cljs");
      Put_Line ("  Prolog family: swi-prolog, gnu-prolog, mercury, prolog (generic)");
      Put_Line ("  Functional/Formal: futhark, lean4, haskell");
   end Print_Usage;

   function Parse_Target (Target_Str : String) return Target_Language is
   begin
      --  Mainstream languages
      if Target_Str = "cpp" or Target_Str = "c++" then
         return Target_CPP;
      elsif Target_Str = "c" then
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
      elsif Target_Str = "typescript" or Target_Str = "ts" then
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
      else
         return Target_Python;  --  Default
      end if;
   end Parse_Target;

   Input_Path  : Path_String := Path_Strings.Null_Bounded_String;
   Output_Path : Path_String := Path_Strings.Null_Bounded_String;
   Target      : Target_Language := Target_Python;
   Status      : Status_Code;

begin
   --  Parse command-line arguments
   if ACL.Argument_Count < 3 then
      Print_Usage;
      ACL.Set_Exit_Status (ACL.Failure);
      return;
   end if;

   declare
      Arg : constant String := ACL.Argument (1);
   begin
      if Arg = "-h" or Arg = "--help" then
         Print_Usage;
         ACL.Set_Exit_Status (ACL.Success);
         return;
      end if;
   end;

   Input_Path := Path_Strings.To_Bounded_String (ACL.Argument (1));
   Target := Parse_Target (ACL.Argument (2));
   Output_Path := Path_Strings.To_Bounded_String (ACL.Argument (3));

   --  Process the IR file
   Put_Line ("Generating target code...");
   Put_Line ("  Input:  " & Path_Strings.To_String (Input_Path));
   Put_Line ("  Output: " & Path_Strings.To_String (Output_Path));

   Emit_Target.Emit_Target_File (Input_Path, Target, Output_Path, Status);

   if Status = STUNIR_Types.Success then
      Put_Line ("Code generation successful.");
      ACL.Set_Exit_Status (ACL.Success);
   else
      Put_Line ("Error: Code generation failed with status " & Status_Code'Image (Status));
      ACL.Set_Exit_Status (ACL.Failure);
   end if;

end Code_Emitter_Main;