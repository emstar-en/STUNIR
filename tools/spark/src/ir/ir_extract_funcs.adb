--  ir_extract_funcs - Extract functions array from IR JSON document
--  Input:  Complete IR JSON (stdin)
--  Output: Functions JSON array (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure IR_Extract_Funcs is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Show_Version  : Boolean := False;
   Show_Help     : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""ir_extract_funcs"",""version"":""0.1.0-alpha""," &
     """description"":""Extract functions array from IR JSON document""," &
     """inputs"":[{""type"":""json"",""source"":""stdin"",""required"":true}]," &
     """outputs"":[{""type"":""json_array"",""source"":""stdout""}]}";

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
             JSON (P) = ASCII.CR) loop P := P + 1; end loop;
      if P > JSON'Last then return ""; end if;
      if JSON (P) /= '[' and then JSON (P) /= '{' then return ""; end if;
      E := P;
      while E <= JSON'Last loop
         if InStr then
            if JSON (E) = '"' and then (E = JSON'First or else JSON (E - 1) /= '\') then
               InStr := False; end if;
         else
            if    JSON (E) = '"'                   then InStr := True;
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

begin
   for I in 1 .. Argument_Count loop
      declare
         Arg : constant String := Argument (I);
      begin
         if    Arg = "--help" or Arg = "-h"    then Show_Help    := True;
         elsif Arg = "--version" or Arg = "-v" then Show_Version := True;
         elsif Arg = "--describe"              then Show_Describe := True;
         end if;
      end;
   end loop;

   if Show_Help then
      Put_Line ("ir_extract_funcs - Extract functions array from IR JSON");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: ir_extract_funcs < ir.json");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("ir_extract_funcs " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   declare
      Content : constant String := Read_Stdin;
      Funcs   : constant String := Get_Block (Content, "functions");
   begin
      if Content'Length = 0 then
         Put_Line (Standard_Error, "ERROR: No input on stdin");
         Set_Exit_Status (Exit_Error); return;
      end if;
      Put_Line (if Funcs'Length > 0 then Funcs else "[]");
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end IR_Extract_Funcs;
