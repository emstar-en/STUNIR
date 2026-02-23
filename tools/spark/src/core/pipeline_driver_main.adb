with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
with Pipeline_Driver;
with Source_Extract;

procedure Pipeline_Driver_Main is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;

   Input_Path  : Path_String;
   Source_Path : Path_String;
   Output_Path : Path_String;
   Module_Name : Identifier_String;
   Source_Lang : Identifier_String := Identifier_Strings.To_Bounded_String ("c");
   Target      : Target_Language := Target_CPP;
   Generate_All : Boolean := False;
   Verbose      : Boolean := False;
   Status       : Status_Code;
   Results      : Pipeline_Driver.Pipeline_Results;
   Enabled      : Pipeline_Driver.Phase_Enabled := [others => False];

   procedure Print_Usage is
   begin
      Put_Line ("Usage: pipeline_driver -i <input.json> -o <output_dir> [-m <module_name>] [--source <input.c>] [--lang <lang>] [--target <lang>] [--all] [--verbose]");
      Put_Line ("  -i <input.json>    Path to extraction.json input file");
      Put_Line ("  --source <input>   Source file to extract into extraction.json");
      Put_Line ("  --lang <lang>      Source language for --source (default: c)");
      Put_Line ("  -o <output_dir>    Output directory");
      Put_Line ("  -m <module_name>   Module name (optional, defaults to 'module')");
      Put_Line ("  --target <lang>    Target language:");
      Put_Line ("                       Mainstream: cpp, c, c99, rust, python, go, js, java,");
      Put_Line ("                                   csharp, swift, kotlin, spark");
      Put_Line ("                       Lisp family: common-lisp, scheme, racket, emacs-lisp,");
      Put_Line ("                                    guile, hy, janet, clojure, cljs");
      Put_Line ("                       Prolog family: swi-prolog, gnu-prolog, mercury, prolog");
      Put_Line ("                       Functional/Formal: futhark, lean4");
      Put_Line ("  --all              Generate all targets");
      Put_Line ("  --verbose          Verbose output");
   end Print_Usage;

   function Parse_Target (S : String) return Target_Language is
   begin
      --  Mainstream languages
      if S = "cpp" or S = "c++" then
         return Target_CPP;
      elsif S = "c" or S = "c99" then
         return Target_C;
      elsif S = "rust" or S = "rs" then
         return Target_Rust;
      elsif S = "python" or S = "py" then
         return Target_Python;
      elsif S = "go" or S = "golang" then
         return Target_Go;
      elsif S = "js" or S = "javascript" then
         return Target_JavaScript;
      elsif S = "java" then
         return Target_Java;
      elsif S = "csharp" or S = "cs" or S = "c#" then
         return Target_CSharp;
      elsif S = "swift" then
         return Target_Swift;
      elsif S = "kotlin" or S = "kt" then
         return Target_Kotlin;
      elsif S = "spark" then
         return Target_SPARK;
      elsif S = "ada" then
         return Target_Ada;
      --  Lisp family
      elsif S = "common-lisp" or S = "cl" or S = "lisp" then
         return Target_Common_Lisp;
      elsif S = "scheme" then
         return Target_Scheme;
      elsif S = "racket" then
         return Target_Racket;
      elsif S = "emacs-lisp" or S = "elisp" or S = "el" then
         return Target_Emacs_Lisp;
      elsif S = "guile" then
         return Target_Guile;
      elsif S = "hy" then
         return Target_Hy;
      elsif S = "janet" then
         return Target_Janet;
      elsif S = "clojure" or S = "clj" then
         return Target_Clojure;
      elsif S = "cljs" or S = "clojurescript" then
         return Target_ClojureScript;
      --  Prolog family
      elsif S = "swi-prolog" or S = "swi" then
         return Target_SWI_Prolog;
      elsif S = "gnu-prolog" or S = "gprolog" then
         return Target_GNU_Prolog;
      elsif S = "mercury" then
         return Target_Mercury;
      elsif S = "prolog" then
         return Target_Prolog;  --  Generic (deprecated)
      --  Functional/Formal
      elsif S = "futhark" then
         return Target_Futhark;
      elsif S = "lean4" or S = "lean" then
         return Target_Lean4;
      elsif S = "haskell" or S = "hs" then
         return Target_Haskell;
      else
         return Target_CPP;
      end if;
   end Parse_Target;

begin
   if Argument_Count < 4 then
      Print_Usage;
      Set_Exit_Status (Failure);
      return;
   end if;

   declare
      I : Positive := 1;
   begin
      while I <= Argument_Count loop
         declare
            Arg : constant String := Argument (I);
         begin
            if Arg = "-i" and I < Argument_Count then
               I := I + 1;
               Input_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "--source" and I < Argument_Count then
               I := I + 1;
               Source_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "-o" and I < Argument_Count then
               I := I + 1;
               Output_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "-m" and I < Argument_Count then
               I := I + 1;
               Module_Name := Identifier_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "--lang" and I < Argument_Count then
               I := I + 1;
               Source_Lang := Identifier_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "--target" and I < Argument_Count then
               I := I + 1;
               Target := Parse_Target (Argument (I));
            elsif Arg = "--all" then
               Generate_All := True;
            elsif Arg = "--verbose" then
               Verbose := True;
            else
               Put_Line (Standard_Error, "Error: Unknown argument: " & Arg);
               Set_Exit_Status (Failure);
               return;
            end if;
            I := I + 1;
         end;
      end loop;
   end;

   if Path_Strings.Length (Input_Path) = 0 and then Path_Strings.Length (Source_Path) = 0 then
      Put_Line (Standard_Error, "Error: Input path is required (-i) or provide --source");
      Set_Exit_Status (Failure);
      return;
   end if;

   if Path_Strings.Length (Output_Path) = 0 then
      Put_Line (Standard_Error, "Error: Output directory is required (-o)");
      Set_Exit_Status (Failure);
      return;
   end if;

   if Identifier_Strings.Length (Module_Name) = 0 then
      Module_Name := Identifier_Strings.To_Bounded_String ("module");
   end if;

   --  If --source provided, run source extractor to produce extraction.json
   if Path_Strings.Length (Source_Path) > 0 then
      declare
         Extract_Path : constant String := Path_Strings.To_String (Output_Path) & "/extraction.json";
         Extract_Status : Status_Code;
      begin
         Source_Extract.Extract_File
           (Input_Path  => Source_Path,
            Output_Path => Path_Strings.To_Bounded_String (Extract_Path),
            Module_Name => Module_Name,
            Language    => Source_Lang,
            Status      => Extract_Status);
         if Extract_Status /= STUNIR_Types.Success then
            Put_Line (Standard_Error, "Error: Source extraction failed: " & Status_Code_Image (Extract_Status));
            Set_Exit_Status (Failure);
            return;
         end if;
         Input_Path := Path_Strings.To_Bounded_String (Extract_Path);
      exception
         when others =>
            Put_Line (Standard_Error, "Error: Source extraction failed");
            Set_Exit_Status (Failure);
            return;
      end;
   end if;

   Enabled (Pipeline_Driver.Phase_Spec_Assembly) := True;
   Enabled (Pipeline_Driver.Phase_IR_Conversion) := True;
   Enabled (Pipeline_Driver.Phase_Code_Emission) := True;

   declare
      Config : Pipeline_Driver.Pipeline_Config := (
         Input_Path     => Input_Path,
         Output_Dir     => Output_Path,
         Module_Name    => Module_Name,
         Enabled_Phases => Enabled,
         Targets        => Target,
         Generate_All   => Generate_All,
         Verbose        => Verbose);
   begin
      Pipeline_Driver.Run_Full_Pipeline (Config, Results, Status);
      Pipeline_Driver.Print_Results (Results, Verbose);
   end;

   if Is_Success (Status) then
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Success);
   else
      Put_Line (Standard_Error, "[ERROR] Pipeline failed: " & Status_Code_Image (Status));
      Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end if;
end Pipeline_Driver_Main;