--  spec_modular_resolve - Resolve modular spec dependencies
--  Input:  directory of spec JSON files + registry JSON
--  Output: dependency-ordered manifest of spec file paths (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Directories;
with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;
with STUNIR_Types;
with Spec_Parse;
with Type_Map_Runtime;

procedure Spec_Modular_Resolve is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Directories;
   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;
   use Ada.Characters.Handling;
   use STUNIR_Types;

   Exit_Success          : constant := 0;
   Exit_Validation_Error : constant := 1;
   Exit_Processing_Error : constant := 2;

   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
       "{" & ASCII.LF &
       "  ""tool"": ""spec_modular_resolve""," & ASCII.LF &
       "  ""version"": ""0.1.0-alpha""," & ASCII.LF &
       "  ""description"": ""Resolve modular spec dependencies and emit an ordered manifest""," & ASCII.LF &
       "  ""inputs"": [" & ASCII.LF &
       "    {""name"": ""spec_dir"", ""type"": ""path"", ""source"": ""argument"", ""required"": true}," & ASCII.LF &
       "    {""name"": ""registry"", ""type"": ""json"", ""source"": ""file"", ""required"": true}" & ASCII.LF &
       "  ]," & ASCII.LF &
       "  ""outputs"": [" & ASCII.LF &
       "    {""name"": ""manifest"", ""type"": ""json"", ""source"": ""stdout""}" & ASCII.LF &
       "  ]," & ASCII.LF &
       "  ""options"": [""--help"", ""--version"", ""--describe"", ""--registry"", ""--pattern""]" & ASCII.LF &
       "}";

   Spec_Dir      : Unbounded_String := Null_Unbounded_String;
   Registry_Path : Unbounded_String := Null_Unbounded_String;
   Pattern       : Unbounded_String := To_Unbounded_String ("*.json");

   Max_Spec_Files     : constant := 512;
   Max_Registry_Types : constant := 4096;

   type String_List is array (Positive range <>) of Unbounded_String;

   Spec_Paths : String_List (1 .. Max_Spec_Files);
   Spec_Keys  : String_List (1 .. Max_Spec_Files);
   Spec_Count : Natural := 0;

   Type_Names    : String_List (1 .. Max_Registry_Types);
   Type_Specs    : String_List (1 .. Max_Registry_Types);
   Type_Count    : Natural := 0;

   type Bool_Matrix is array (Positive range <>, Positive range <>) of Boolean;
   Adj : Bool_Matrix (1 .. Max_Spec_Files, 1 .. Max_Spec_Files) := (others => (others => False));
   Indegree : array (1 .. Max_Spec_Files) of Natural := (others => 0);
   Emitted  : array (1 .. Max_Spec_Files) of Boolean := (others => False);
   Order    : array (1 .. Max_Spec_Files) of Natural := (others => 0);

   procedure Print_Usage is
   begin
      Put_Line ("spec_modular_resolve - Resolve modular spec dependencies");
      Put_Line ("Version: " & Version);
      Put_Line ("");
      Put_Line ("Usage: spec_modular_resolve <spec_dir> --registry <registry.json> [options]");
      Put_Line ("");
      Put_Line ("Arguments:");
      Put_Line ("  <spec_dir>        Directory containing spec JSON files");
      Put_Line ("");
      Put_Line ("Options:");
      Put_Line ("  --help, -h        Show this help message");
      Put_Line ("  --version, -v     Show version information");
      Put_Line ("  --describe        Show tool description (JSON)");
      Put_Line ("  --registry FILE   Registry JSON path (required)");
      Put_Line ("  --pattern GLOB    File glob pattern (default: *.json)");
      Put_Line ("");
      Put_Line ("Description:");
      Put_Line ("  Reads spec JSON files from <spec_dir>, resolves cross-spec type");
      Put_Line ("  dependencies using the registry, and emits a dependency-ordered");
      Put_Line ("  manifest to stdout. The manifest lists spec files in topological");
      Put_Line ("  order so downstream tools can process them correctly.");
      Put_Line ("");
      Put_Line ("Registry format:");
      Put_Line ("  {""types"": [{""name"": ""TypeName"", ""spec"": ""path/to/spec.json""}, ...]}");
      Put_Line ("");
      Put_Line ("Output format:");
      Put_Line ("  JSON manifest with ordered ""specs"" array");
      Put_Line ("");
      Put_Line ("Exit codes:");
      Put_Line ("  0 = success");
      Put_Line ("  1 = validation error (missing args, type conflicts)");
      Put_Line ("  2 = processing error (parse failure, cyclic deps)");
   end Print_Usage;

   procedure Print_Error (Msg : String) is
   begin
      Put_Line (Standard_Error, "ERROR: " & Msg);
   end Print_Error;

   function Read_File (Path : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
      F      : Ada.Text_IO.File_Type;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      Ada.Text_IO.Open (F, Ada.Text_IO.In_File, Path);
      while not Ada.Text_IO.End_Of_File (F) loop
         Ada.Text_IO.Get_Line (F, Line, Last);
         if Last > 0 then
            Append (Result, Line (1 .. Last));
         end if;
         if not Ada.Text_IO.End_Of_File (F) then
            Append (Result, ASCII.LF);
         end if;
      end loop;
      Ada.Text_IO.Close (F);
      return To_String (Result);
   exception
      when others =>
         return "";
   end Read_File;

   function Normalize_Key (S : String) return String is
      Result : Unbounded_String := Null_Unbounded_String;
   begin
      for C of S loop
         if C = '/' then
            Append (Result, "\");
         else
            Append (Result, To_Lower (C));
         end if;
      end loop;
      return To_String (Result);
   end Normalize_Key;

   function Is_Absolute (Path : String) return Boolean is
   begin
      if Path'Length = 0 then
         return False;
      end if;
      if Path'Length >= 2 and then Path (Path'First + 1) = ':' then
         return True;
      end if;
         return Path (Path'First) = '/' or else Path (Path'First) = '\';
   end Is_Absolute;

   function Join_Path (Base, Rel : String) return String is
   begin
      if Base'Length = 0 then
         return Rel;
      end if;
      if Base (Base'Last) = '/' or else Base (Base'Last) = '\' then
         return Base & Rel;
      end if;
      return Base & '/' & Rel;
   end Join_Path;

   function Normalize_Path (Base, Path : String) return String is
   begin
      if Is_Absolute (Path) then
         return Path;
      end if;
      return Join_Path (Base, Path);
   end Normalize_Path;

   function Find (S : String; P : String) return Natural is
   begin
      if P'Length = 0 or else P'Length > S'Length then
         return 0;
      end if;
      for I in S'First .. S'Last - P'Length + 1 loop
         if S (I .. I + P'Length - 1) = P then
            return I;
         end if;
      end loop;
      return 0;
   end Find;

   function Get_String (JSON : String; Key : String) return String is
      Pat  : constant String := """" & Key & """";
      K    : constant Natural := Find (JSON, Pat);
      P, E : Natural;
   begin
      if K = 0 then
         return "";
      end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or JSON (P) = ASCII.HT or
             JSON (P) = ASCII.LF or JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last or else JSON (P) /= '"' then
         return "";
      end if;
      P := P + 1;
      E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop
         E := E + 1;
      end loop;
      if E > JSON'Last then
         return JSON (P .. JSON'Last);
      end if;
      return JSON (P .. E - 1);
   end Get_String;

   function Get_Block (JSON : String; Key : String) return String is
      Pat   : constant String := """" & Key & """";
      K     : constant Natural := Find (JSON, Pat);
      P, E  : Natural;
      Depth : Integer := 0;
      InStr : Boolean := False;
   begin
      if K = 0 then
         return "";
      end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or JSON (P) = ASCII.HT or
             JSON (P) = ASCII.LF or JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last then
         return "";
      end if;
      if JSON (P) /= '[' and then JSON (P) /= '{' then
         return "";
      end if;
      E := P;
      while E <= JSON'Last loop
         if InStr then
            if JSON (E) = '"' and then (E = JSON'First or else JSON (E - 1) /= '\') then
               InStr := False;
            end if;
         else
            if JSON (E) = '"' then
               InStr := True;
            elsif JSON (E) = '{' or JSON (E) = '[' then
               Depth := Depth + 1;
            elsif JSON (E) = '}' or JSON (E) = ']' then
               Depth := Depth - 1;
               if Depth = 0 then
                  return JSON (P .. E);
               end if;
            end if;
         end if;
         E := E + 1;
      end loop;
      return "";
   end Get_Block;

   function Get_Element (Arr : String; N : Natural) return String is
      Pos   : Natural := Arr'First;
      Cnt   : Natural := 0;
      Depth : Integer := 0;
      InStr : Boolean := False;
      ElS   : Natural := 0;
   begin
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop
         Pos := Pos + 1;
      end loop;
      if Pos > Arr'Last then
         return "";
      end if;
      Pos := Pos + 1;
      while Pos <= Arr'Last loop
         if InStr then
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' => if Depth = 0 then ElS := Pos; end if; Depth := Depth + 1;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  if Depth = 0 and ElS > 0 then
                     if Cnt = N then
                        return Arr (ElS .. Pos);
                     end if;
                     Cnt := Cnt + 1;
                     ElS := 0;
                  end if;
               when others => null;
            end case;
         end if;
         Pos := Pos + 1;
      end loop;
      return "";
   end Get_Element;

   function Count_Elements (Arr : String) return Natural is
      Pos   : Natural := Arr'First;
      Cnt   : Natural := 0;
      Depth : Integer := 0;
      InStr : Boolean := False;
   begin
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop
         Pos := Pos + 1;
      end loop;
      if Pos > Arr'Last then
         return 0;
      end if;
      Pos := Pos + 1;
      while Pos <= Arr'Last loop
         if InStr then
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' => if Depth = 0 then Cnt := Cnt + 1; end if; Depth := Depth + 1;
               when '}' | ']' => Depth := Depth - 1;
               when others    => null;
            end case;
         end if;
         Pos := Pos + 1;
      end loop;
      return Cnt;
   end Count_Elements;

   function Base_Type_Name (Type_Name : String) return String is
      Trimmed : constant String := Trim (Type_Name, Ada.Strings.Both);
      Last    : Natural := 0;
   begin
      if Trimmed'Length = 0 then
         return "";
      end if;
      for I in Trimmed'Range loop
         if Is_Alphanumeric (Trimmed (I)) or else Trimmed (I) = '_' or else Trimmed (I) = '.' then
            Last := I;
         else
            exit;
         end if;
      end loop;
      if Last = 0 then
         return "";
      end if;
      return Trimmed (Trimmed'First .. Last);
   end Base_Type_Name;

   function Find_Spec_Index (Path_Key : String) return Natural is
   begin
      for I in 1 .. Spec_Count loop
         if To_String (Spec_Keys (I)) = Path_Key then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Spec_Index;

   function Find_Type_Spec (Type_Name : String) return String is
   begin
      for I in 1 .. Type_Count loop
         if To_String (Type_Names (I)) = Type_Name then
            return To_String (Type_Specs (I));
         end if;
      end loop;
      return "";
   end Find_Type_Spec;

   procedure Register_Type (Name : String; Spec_Path : String; Ok : in out Boolean) is
   begin
      for I in 1 .. Type_Count loop
         if To_String (Type_Names (I)) = Name then
            if To_String (Type_Specs (I)) /= Spec_Path then
               Print_Error ("Type conflict: " & Name & " defined in multiple specs");
               Ok := False;
            end if;
            return;
         end if;
      end loop;
      if Type_Count >= Max_Registry_Types then
         Print_Error ("Registry type limit exceeded");
         Ok := False;
         return;
      end if;
      Type_Count := Type_Count + 1;
      Type_Names (Type_Count) := To_Unbounded_String (Name);
      Type_Specs (Type_Count) := To_Unbounded_String (Spec_Path);
   end Register_Type;

   procedure Add_Spec (Path : String; Ok : in out Boolean) is
      Full : constant String := Full_Name (Path);
      Key  : constant String := Normalize_Key (Full);
   begin
      if Spec_Count >= Max_Spec_Files then
         Print_Error ("Spec file limit exceeded");
         Ok := False;
         return;
      end if;
      if Find_Spec_Index (Key) /= 0 then
         return;
      end if;
      Spec_Count := Spec_Count + 1;
      Spec_Paths (Spec_Count) := To_Unbounded_String (Full);
      Spec_Keys (Spec_Count) := To_Unbounded_String (Key);
   end Add_Spec;

   procedure Add_Edge (From_Idx, To_Idx : Natural) is
   begin
      if From_Idx = 0 or else To_Idx = 0 then
         return;
      end if;
      if not Adj (From_Idx, To_Idx) then
         Adj (From_Idx, To_Idx) := True;
         Indegree (To_Idx) := Indegree (To_Idx) + 1;
      end if;
   end Add_Edge;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if Arg = "--help" or Arg = "-h" then
            Show_Help := True;
         elsif Arg = "--version" or Arg = "-v" then
            Show_Version := True;
         elsif Arg = "--describe" then
            Show_Describe := True;
         elsif Arg'Length > 11 and then Arg (1 .. 11) = "--registry=" then
            Registry_Path := To_Unbounded_String (Arg (12 .. Arg'Last));
         elsif Arg = "--registry" and then I < Argument_Count then
            Registry_Path := To_Unbounded_String (Argument (I + 1));
         elsif Arg'Length > 10 and then Arg (1 .. 10) = "--pattern=" then
            Pattern := To_Unbounded_String (Arg (11 .. Arg'Last));
         elsif Arg = "--pattern" and then I < Argument_Count then
            Pattern := To_Unbounded_String (Argument (I + 1));
         elsif Arg'Length > 0 and then Arg (Arg'First) /= '-' then
            if Length (Spec_Dir) = 0 then
               Spec_Dir := To_Unbounded_String (Arg);
            end if;
         end if;
      end;
   end loop;

   if Show_Help then
      Print_Usage;
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Version then
      Put_Line ("spec_modular_resolve " & Version);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success);
      return;
   end if;

   if Length (Spec_Dir) = 0 or else Length (Registry_Path) = 0 then
      Print_Error ("spec_dir and --registry are required");
      Set_Exit_Status (Exit_Validation_Error);
      return;
   end if;

   declare
      Ok          : Boolean := True;
      Spec_Dir_F  : constant String := Full_Name (To_String (Spec_Dir));
      Registry_F  : constant String := Full_Name (To_String (Registry_Path));
      Registry    : constant String := Read_File (Registry_F);
      Types_Block : constant String := Get_Block (Registry, "types");
      Type_N      : constant Natural := Count_Elements (Types_Block);
   begin
      if Registry'Length = 0 then
         Print_Error ("Failed to read registry file");
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      --  Load registry type mapping
      for I in 0 .. Type_N - 1 loop
         declare
            Obj  : constant String := Get_Element (Types_Block, I);
            Name : constant String := Get_String (Obj, "name");
            Spec : constant String := Get_String (Obj, "spec");
            Norm : constant String := Normalize_Path (Spec_Dir_F, Spec);
            Key  : constant String := Normalize_Key (Norm);
         begin
            if Name'Length = 0 or else Spec'Length = 0 then
               Print_Error ("Invalid registry entry in types array");
               Ok := False;
            else
               Register_Type (Name, Norm, Ok);
               if Find_Spec_Index (Key) = 0 then
                  Add_Spec (Norm, Ok);
               end if;
            end if;
         end;
      end loop;

      --  Build type field dependencies from registry
      --  Each type entry can have a "fields" array with field types
      for I in 0 .. Type_N - 1 loop
         declare
            Obj          : constant String := Get_Element (Types_Block, I);
            TypeName     : constant String := Get_String (Obj, "name");
            TypeSpec     : constant String := Find_Type_Spec (TypeName);
            TypeSpecKey  : constant String := Normalize_Key (TypeSpec);
            TypeIdx      : constant Natural := Find_Spec_Index (TypeSpecKey);
            Fields_Block : constant String := Get_Block (Obj, "fields");
            Fields_N     : constant Natural := Count_Elements (Fields_Block);
         begin
            if TypeIdx = 0 then
               goto Continue_Field_Loop;
            end if;

            --  Process each field's type
            for J in 0 .. Fields_N - 1 loop
               declare
                  Field_Obj  : constant String := Get_Element (Fields_Block, J);
                  Field_Type : constant String := Base_Type_Name (Get_String (Field_Obj, "type"));
               begin
                  if Field_Type'Length > 0 and then not Type_Map_Runtime.Is_Primitive_Type (Field_Type) then
                     declare
                        Owner      : constant String := Find_Type_Spec (Field_Type);
                        Owner_Key  : constant String := Normalize_Key (Owner);
                        Owner_Idx  : constant Natural := Find_Spec_Index (Owner_Key);
                     begin
                        if Owner'Length > 0 and then Owner_Idx > 0 and then Owner_Idx /= TypeIdx then
                           Add_Edge (Owner_Idx, TypeIdx);
                        end if;
                     end;
                  end if;
               end;
            end loop;

            <<Continue_Field_Loop>>
         end;
      end loop;

      --  Enumerate spec files in directory
      declare
         Search      : Search_Type;
         Dir_Entry   : Directory_Entry_Type;
         Pattern_Str : constant String := To_String (Pattern);
      begin
         Start_Search (Search, Spec_Dir_F, Pattern_Str);
         while More_Entries (Search) loop
            Get_Next_Entry (Search, Dir_Entry);
            if Kind (Dir_Entry) = Ordinary_File then
               Add_Spec (Full_Name (Dir_Entry), Ok);
            end if;
         end loop;
         End_Search (Search);
      exception
         when others =>
            Print_Error ("Failed to search spec directory");
            Ok := False;
      end;

      if not Ok then
         Set_Exit_Status (Exit_Processing_Error);
         return;
      end if;

      if Spec_Count = 0 then
         Print_Error ("No spec files found");
         Set_Exit_Status (Exit_Validation_Error);
         return;
      end if;

      --  Build dependency graph from spec usage
      for I in 1 .. Spec_Count loop
         declare
            Spec_Path : constant String := To_String (Spec_Paths (I));
            Spec_Data : Spec_Parse.Spec_Data;
            Status    : Status_Code;
         begin
            Spec_Parse.Parse_Spec_File (Path_Strings.To_Bounded_String (Spec_Path), Spec_Data, Status);
            if Status /= STUNIR_Types.Success then
               Print_Error ("Failed to parse spec: " & Spec_Path);
               Set_Exit_Status (Exit_Processing_Error);
               return;
            end if;

            for F in 1 .. Spec_Data.Functions.Count loop
               declare
                  Ret_T : constant String := Base_Type_Name (Type_Name_Strings.To_String (Spec_Data.Functions.Functions (F).Return_Type));
               begin
                  if Ret_T'Length > 0 and then not Type_Map_Runtime.Is_Primitive_Type (Ret_T) then
                     declare
                        Owner : constant String := Find_Type_Spec (Ret_T);
                        Owner_Key : constant String := Normalize_Key (Owner);
                        Owner_Idx : constant Natural := Find_Spec_Index (Owner_Key);
                     begin
                        if Owner'Length = 0 then
                           Print_Error ("Type not in registry: " & Ret_T);
                           Set_Exit_Status (Exit_Processing_Error);
                           return;
                        end if;
                        if Owner_Idx = 0 then
                           Print_Error ("Registry spec missing from spec set: " & Owner);
                           Set_Exit_Status (Exit_Processing_Error);
                           return;
                        end if;
                        if Owner_Idx /= I then
                           Add_Edge (Owner_Idx, I);
                        end if;
                     end;
                  end if;
               end;

               for P in 1 .. Spec_Data.Functions.Functions (F).Parameters.Count loop
                  declare
                     Param_T : constant String := Base_Type_Name (
                       Type_Name_Strings.To_String (Spec_Data.Functions.Functions (F).Parameters.Params (P).Param_Type));
                  begin
                     if Param_T'Length > 0 and then not Type_Map_Runtime.Is_Primitive_Type (Param_T) then
                        declare
                           Owner : constant String := Find_Type_Spec (Param_T);
                           Owner_Key : constant String := Normalize_Key (Owner);
                           Owner_Idx : constant Natural := Find_Spec_Index (Owner_Key);
                        begin
                           if Owner'Length = 0 then
                              Print_Error ("Type not in registry: " & Param_T);
                              Set_Exit_Status (Exit_Processing_Error);
                              return;
                           end if;
                           if Owner_Idx = 0 then
                              Print_Error ("Registry spec missing from spec set: " & Owner);
                              Set_Exit_Status (Exit_Processing_Error);
                              return;
                           end if;
                           if Owner_Idx /= I then
                              Add_Edge (Owner_Idx, I);
                           end if;
                        end;
                     end if;
                  end;
               end loop;
            end loop;
         end;
      end loop;

      --  Topological ordering
      for K in 1 .. Spec_Count loop
         declare
            Candidate : Natural := 0;
         begin
            for I in 1 .. Spec_Count loop
               if not Emitted (I) and then Indegree (I) = 0 then
                  if Candidate = 0 or else To_String (Spec_Keys (I)) < To_String (Spec_Keys (Candidate)) then
                     Candidate := I;
                  end if;
               end if;
            end loop;

            if Candidate = 0 then
               Print_Error ("Cyclic dependency detected in specs");
               Set_Exit_Status (Exit_Processing_Error);
               return;
            end if;

            Emitted (Candidate) := True;
            Order (K) := Candidate;

            for J in 1 .. Spec_Count loop
               if Adj (Candidate, J) then
                  if Indegree (J) > 0 then
                     Indegree (J) := Indegree (J) - 1;
                  end if;
               end if;
            end loop;
         end;
      end loop;

      --  Emit manifest JSON
      Put_Line ("{");
      Put_Line ("  ""tool"": ""spec_modular_resolve"",");
      Put_Line ("  ""version"": """ & Version & """,");
      Put_Line ("  ""spec_dir"": """ & Spec_Dir_F & """,");
      Put_Line ("  ""registry"": """ & Registry_F & """,");
      Put_Line ("  ""specs"": [");
      for I in 1 .. Spec_Count loop
         declare
            Idx : constant Natural := Order (I);
            S   : constant String := To_String (Spec_Paths (Idx));
         begin
            if I < Spec_Count then
               Put_Line ("    """ & S & """,");
            else
               Put_Line ("    """ & S & """");
            end if;
         end;
      end loop;
      Put_Line ("  ]");
      Put_Line ("}");
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Print_Error ("Unexpected error");
      Set_Exit_Status (Exit_Processing_Error);
end Spec_Modular_Resolve;
