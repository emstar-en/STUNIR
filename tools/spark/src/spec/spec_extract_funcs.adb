--  spec_extract_funcs - Extract functions array from STUNIR spec JSON
--  Input:  Spec JSON (stdin)
--  Output: Functions JSON array (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure Spec_Extract_Funcs is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Module_Filter : Unbounded_String := Null_Unbounded_String;
   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""spec_extract_funcs"",""version"":""0.1.0-alpha""," &
     """description"":""Extract functions array from STUNIR spec JSON""," &
     """inputs"":[{""type"":""json"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json_array"",""source"":""stdout""}]," &
     """options"":[""--help"",""--version"",""--describe"",""--module NAME""]}";

   function Read_Stdin return String is
      Result : Unbounded_String := Null_Unbounded_String;
      Line   : String (1 .. 4096);
      Last   : Natural;
   begin
      while not End_Of_File loop
         Get_Line (Line, Last);
         Append (Result, Line (1 .. Last));
         Append (Result, ASCII.LF);
      end loop;
      return To_String (Result);
   end Read_Stdin;

   function Find (S : String; P : String) return Natural is
   begin
      if P'Length = 0 or P'Length > S'Length then return 0; end if;
      for I in S'First .. S'Last - P'Length + 1 loop
         if S (I .. I + P'Length - 1) = P then return I; end if;
      end loop;
      return 0;
   end Find;

   function Get_String (JSON : String; Key : String) return String is
      Pat : constant String := """" & Key & """";
      K   : constant Natural := Find (JSON, Pat);
      P, E : Natural;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last or else JSON (P) /= '"' then return ""; end if;
      P := P + 1; E := P;
      while E <= JSON'Last and then JSON (E) /= '"' loop E := E + 1; end loop;
      if E > JSON'Last then return JSON (P .. JSON'Last); end if;
      return JSON (P .. E - 1);
   end Get_String;

   function Get_Block (JSON : String; Key : String) return String is
      Pat   : constant String := """" & Key & """";
      K     : constant Natural := Find (JSON, Pat);
      P, E  : Natural;
      Depth : Integer := 0;
      InStr : Boolean := False;
   begin
      if K = 0 then return ""; end if;
      P := K + Pat'Length;
      while P <= JSON'Last and then
            (JSON (P) = ':' or JSON (P) = ' ' or
             JSON (P) = ASCII.HT or JSON (P) = ASCII.LF or
             JSON (P) = ASCII.CR) loop
         P := P + 1;
      end loop;
      if P > JSON'Last then return ""; end if;
      if JSON (P) /= '[' and then JSON (P) /= '{' then return ""; end if;
      E := P;
      while E <= JSON'Last loop
         if InStr then
            if JSON (E) = '"' and then (E = JSON'First or else JSON (E - 1) /= '\') then
               InStr := False;
            end if;
         else
            if    JSON (E) = '"'             then InStr := True;
            elsif JSON (E) = '{' or JSON (E) = '[' then Depth := Depth + 1;
            elsif JSON (E) = '}' or JSON (E) = ']' then
               Depth := Depth - 1;
               if Depth = 0 then return JSON (P .. E); end if;
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
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop Pos := Pos + 1; end loop;
      if Pos > Arr'Last then return ""; end if;
      Pos := Pos + 1;
      while Pos <= Arr'Last loop
         if InStr then
            if Arr (Pos) = '"' and then (Pos = Arr'First or else Arr (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Arr (Pos) is
               when '"'       => InStr := True;
               when '{' | '[' =>
                  if Depth = 0 then ElS := Pos; end if;
                  Depth := Depth + 1;
               when '}' | ']' =>
                  Depth := Depth - 1;
                  if Depth = 0 and ElS > 0 then
                     if Cnt = N then return Arr (ElS .. Pos); end if;
                     Cnt := Cnt + 1; ElS := 0;
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
      while Pos <= Arr'Last and then Arr (Pos) /= '[' loop Pos := Pos + 1; end loop;
      if Pos > Arr'Last then return 0; end if;
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

   function Get_Module (Spec : String; Mod_Name : String) return String is
      Modules : constant String := Get_Block (Spec, "modules");
      N       : constant Natural := Count_Elements (Modules);
   begin
      for I in 0 .. N - 1 loop
         declare
            Obj : constant String := Get_Element (Modules, I);
            Nam : constant String := Get_String (Obj, "name");
         begin
            if Mod_Name = "" or else Nam = Mod_Name then return Obj; end if;
         end;
      end loop;
      return "";
   end Get_Module;

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if    Arg = "--help" or Arg = "-h"    then Show_Help    := True;
         elsif Arg = "--version" or Arg = "-v" then Show_Version := True;
         elsif Arg = "--describe"              then Show_Describe := True;
         elsif Arg'Length > 9 and then Arg (1 .. 9) = "--module=" then
            Module_Filter := To_Unbounded_String (Arg (10 .. Arg'Last));
         elsif Arg = "--module" and I < Argument_Count then
            Module_Filter := To_Unbounded_String (Argument (I + 1));
         end if;
      end;
   end loop;

   if Show_Help then
      Put_Line ("spec_extract_funcs - Extract functions array from STUNIR spec JSON");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: spec_extract_funcs [OPTIONS] < spec.json");
      Put_Line ("  --module NAME   Extract from named module (default: first module)");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("spec_extract_funcs " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   declare
      Spec       : constant String := Read_Stdin;
      Module_Obj : constant String := Get_Module (Spec, To_String (Module_Filter));
      Funcs      : constant String := Get_Block (Module_Obj, "functions");
   begin
      if Spec'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      if Module_Obj'Length = 0 then
         Put_Line (Standard_Error, "ERROR: Module not found in spec");
         Set_Exit_Status (Exit_Error); return;
      end if;
      Put_Line (if Funcs'Length > 0 then Funcs else "[]");
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end Spec_Extract_Funcs;
