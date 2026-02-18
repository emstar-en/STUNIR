with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Directories;

procedure Lang_Detect is
   use Ada.Text_IO;
   use Ada.Command_Line;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;
   use Ada.Directories;

   procedure Print_Usage is
   begin
      Put_Line (Standard_Error, "Usage: lang_detect [options] [file]");
      Put_Line (Standard_Error, "");
      Put_Line (Standard_Error, "Options:");
      Put_Line (Standard_Error, "  --json           Output result as JSON");
      Put_Line (Standard_Error, "  --confidence     Show confidence score");
      Put_Line (Standard_Error, "  --content        Force content analysis");
      Put_Line (Standard_Error, "  --help           Show this help");
      Put_Line (Standard_Error, "  --version        Show version");
      Put_Line (Standard_Error, "  --describe       Show AI introspection data");
   end Print_Usage;

   procedure Print_Describe is
   begin
      Put_Line ("{");
      Put_Line ("  ""name"": ""lang_detect"",");
      Put_Line ("  ""description"": ""Detect programming language from file extension or content"",");
      Put_Line ("  ""version"": ""0.1.0-alpha"",");
      Put_Line ("  ""inputs"": [");
      Put_Line ("    {""name"": ""file"", ""type"": ""argument"", ""description"": ""File to analyze""}");
      Put_Line ("  ],");
      Put_Line ("  ""outputs"": [");
      Put_Line ("    {""type"": ""string"", ""description"": ""Language ID (c, cpp, rust, python, etc.)""}");
      Put_Line ("  ],");
      Put_Line ("  ""options"": [");
      Put_Line ("    {""name"": ""--json"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--confidence"", ""type"": ""boolean""},");
      Put_Line ("    {""name"": ""--content"", ""type"": ""boolean""}");
      Put_Line ("  ]");
      Put_Line ("}");
   end Print_Describe;

   File_Path   : Unbounded_String := Null_Unbounded_String;
   Output_Json : Boolean := False;
   Show_Conf   : Boolean := False;
   Check_Cont  : Boolean := False;

   function To_Lower (S : String) return String is
      Result : String (S'Range);
   begin
      for I in S'Range loop
         if S (I) in 'A' .. 'Z' then
            Result (I) := Character'Val (Character'Pos (S (I)) + 32);
         else
            Result (I) := S (I);
         end if;
      end loop;
      return Result;
   end To_Lower;

   function Get_Extension (Path : String) return String is
      Idx : Natural := 0;
   begin
      for I in reverse Path'Range loop
         if Path (I) = '.' then
            Idx := I;
            exit;
         elsif Path (I) = '/' or else Path (I) = '\' then
            exit;
         end if;
      end loop;
      
      if Idx > 0 then
         return To_Lower (Path (Idx .. Path'Last));
      else
         return "";
      end if;
   end Get_Extension;

   function Analyze_Content (Path : String) return String is
      --  Simple heuristic scan
      File : File_Type;
      Line : String (1 .. 1024);
      Last : Natural;
      Is_Cpp : Boolean := False;
   begin
      if not Exists (Path) then return ""; end if;
      Open (File, In_File, Path);
      for I in 1 .. 50 loop -- Check first 50 lines
         if End_Of_File (File) then exit; end if;
         Get_Line (File, Line, Last);
         declare
            Line_Str : String := Line (1 .. Last);
         begin
            if Index (Line_Str, "class ") > 0 or else
               Index (Line_Str, "namespace ") > 0 or else
               Index (Line_Str, "template<") > 0 or else
               Index (Line_Str, "std::") > 0 then
               Is_Cpp := True;
            end if;
         end;
      end loop;
      Close (File);

      if Is_Cpp then return "cpp"; else return "c"; end if;
   exception
      when others => 
         if Is_Open (File) then Close (File); end if;
         return "";
   end Analyze_Content;

   Detected_Lang : Unbounded_String := Null_Unbounded_String;
   Confidence    : Float := 0.0;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" then
            Print_Usage; return;
         elsif Arg = "--version" then
            Put_Line ("lang_detect 1.0.0"); return;
         elsif Arg = "--describe" then
            Print_Describe; return;
         elsif Arg = "--json" then
            Output_Json := True;
         elsif Arg = "--confidence" then
            Show_Conf := True;
         elsif Arg = "--content" then
            Check_Cont := True;
         elsif Arg (1) /= '-' then
            File_Path := To_Unbounded_String (Arg);
         end if;
      end;
   end loop;

   if File_Path = Null_Unbounded_String then
      Put_Line (Standard_Error, "Error: No file specified");
      Set_Exit_Status (Failure);
      return;
   end if;

   declare
      Ext : constant String := Get_Extension (To_String (File_Path));
   begin
      if Ext = ".c" then
         Detected_Lang := To_Unbounded_String ("c"); Confidence := 1.0;
      elsif Ext = ".cpp" or else Ext = ".cc" or else Ext = ".cxx" or else Ext = ".hpp" then
         Detected_Lang := To_Unbounded_String ("cpp"); Confidence := 1.0;
      elsif Ext = ".h" then
         -- Ambiguous, check content if requested or default to c/header
         if Check_Cont then
            declare 
               Res : String := Analyze_Content (To_String (File_Path));
            begin
               if Res /= "" then
                  Detected_Lang := To_Unbounded_String (Res);
                  Confidence := 0.8;
               else
                   Detected_Lang := To_Unbounded_String ("c_header");
                   Confidence := 0.5;
               end if;
            end;
         else
            Detected_Lang := To_Unbounded_String ("c_header"); -- Ambiguous
            Confidence := 0.5;
         end if;
      elsif Ext = ".rs" then
         Detected_Lang := To_Unbounded_String ("rust"); Confidence := 1.0;
      elsif Ext = ".py" then
         Detected_Lang := To_Unbounded_String ("python"); Confidence := 1.0;
      elsif Ext = ".adb" or else Ext = ".ads" then
         Detected_Lang := To_Unbounded_String ("ada"); Confidence := 1.0;
      elsif Ext = ".go" then
         Detected_Lang := To_Unbounded_String ("go"); Confidence := 1.0;
      else
         Detected_Lang := To_Unbounded_String ("unknown"); Confidence := 0.0;
      end if;
   end;

   if Output_Json then
      Put ("{""file"": """ & To_String (File_Path) & """, ""language"": """ & To_String (Detected_Lang) & """");
      if Show_Conf then
         Put (", ""confidence"": " & Float'Image (Confidence));
      end if;
      Put_Line ("}");
   else
      Put_Line (To_String (Detected_Lang));
   end if;

end Lang_Detect;
