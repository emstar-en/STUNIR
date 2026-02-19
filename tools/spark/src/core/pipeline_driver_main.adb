with Ada.Command_Line;
with Ada.Text_IO;
with STUNIR_Types;
with Pipeline_Driver;

procedure Pipeline_Driver_Main is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use STUNIR_Types;

   Input_Path  : Path_String;
   Output_Path : Path_String;
   Module_Name : Identifier_String;
   Target      : Target_Language := Target_CPP;
   Generate_All : Boolean := False;
   Verbose      : Boolean := False;
   Status       : Status_Code;
   Results      : Pipeline_Driver.Pipeline_Results;
   Enabled      : Pipeline_Driver.Phase_Enabled := (others => False);

   procedure Print_Usage is
   begin
      Put_Line ("Usage: pipeline_driver -i <input.json> -o <output_dir> [-m <module_name>] [--target <lang>] [--all] [--verbose]");
      Put_Line ("  -i <input.json>    Path to extraction.json input file");
      Put_Line ("  -o <output_dir>    Output directory");
      Put_Line ("  -m <module_name>   Module name (optional, defaults to 'module')");
      Put_Line ("  --target <lang>    Target language (cpp|c|rust|python|go|js|java|csharp|swift|kotlin|spark)");
      Put_Line ("  --all              Generate all targets");
      Put_Line ("  --verbose          Verbose output");
   end Print_Usage;

   function Parse_Target (S : String) return Target_Language is
   begin
      if S = "cpp" then
         return Target_CPP;
      elsif S = "c" then
         return Target_C;
      elsif S = "rust" then
         return Target_Rust;
      elsif S = "python" then
         return Target_Python;
      elsif S = "go" then
         return Target_Go;
      elsif S = "js" then
         return Target_JavaScript;
      elsif S = "java" then
         return Target_Java;
      elsif S = "csharp" then
         return Target_CSharp;
      elsif S = "swift" then
         return Target_Swift;
      elsif S = "kotlin" then
         return Target_Kotlin;
      elsif S = "spark" then
         return Target_SPARK;
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
            elsif Arg = "-o" and I < Argument_Count then
               I := I + 1;
               Output_Path := Path_Strings.To_Bounded_String (Argument (I));
            elsif Arg = "-m" and I < Argument_Count then
               I := I + 1;
               Module_Name := Identifier_Strings.To_Bounded_String (Argument (I));
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

   if Path_Strings.Length (Input_Path) = 0 then
      Put_Line (Standard_Error, "Error: Input path is required (-i)");
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