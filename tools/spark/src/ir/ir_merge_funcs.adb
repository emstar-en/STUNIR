--  ir_merge_funcs - Merge multiple IR function arrays into one
--  Input:  Multiple JSON arrays on stdin (concatenated)
--  Output: Single merged JSON array (stdout)

pragma SPARK_Mode (Off);

with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

procedure IR_Merge_Funcs is
   use Ada.Command_Line;
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   Exit_Success : constant := 0;
   Exit_Error   : constant := 1;

   Show_Help     : Boolean := False;
   Show_Version  : Boolean := False;
   Show_Describe : Boolean := False;

   Version : constant String := "0.1.0-alpha";

   Describe_Output : constant String :=
     "{""tool"":""ir_merge_funcs"",""version"":""0.1.0-alpha""," &
     """description"":""Merge multiple IR function arrays into one""," &
     """inputs"":[{""type"":""json_arrays"",""source"":""stdin"",""required"":true}]," &
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
               when '{' | '[' => if Depth = 0 then ElS := Pos; end if; Depth := Depth + 1;
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

   --  Extract all JSON arrays from input and merge their elements
   function Merge_Arrays (Input : String) return String is
      Pos   : Natural := Input'First;
      Depth : Integer := 0;
      InStr : Boolean := False;
      ArrS  : Natural := 0;
      First : Boolean := True;
      Result : Unbounded_String;
   begin
      Append (Result, "[");
      while Pos <= Input'Last loop
         if InStr then
            if Input (Pos) = '"' and then (Pos = Input'First or else Input (Pos - 1) /= '\') then
               InStr := False;
            end if;
         else
            case Input (Pos) is
               when '"' => InStr := True;
               when '[' =>
                  if Depth = 0 then ArrS := Pos; end if;
                  Depth := Depth + 1;
               when '{' => Depth := Depth + 1;
               when ']' =>
                  Depth := Depth - 1;
                  if Depth = 0 and ArrS > 0 then
                     declare
                        Arr  : constant String  := Input (ArrS .. Pos);
                        N_El : constant Natural := Count_Elements (Arr);
                     begin
                        for I in 0 .. N_El - 1 loop
                           declare
                              El : constant String := Get_Element (Arr, I);
                           begin
                              if El'Length > 0 then
                                 if not First then Append (Result, ","); end if;
                                 Append (Result, El);
                                 First := False;
                              end if;
                           end;
                        end loop;
                     end;
                     ArrS := 0;
                  end if;
               when '}' => Depth := Depth - 1;
               when others => null;
            end case;
         end if;
         Pos := Pos + 1;
      end loop;
      Append (Result, "]");
      return To_String (Result);
   end Merge_Arrays;

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
      Put_Line ("ir_merge_funcs - Merge multiple IR function arrays into one");
      Put_Line ("Version: " & Version);
      New_Line;
      Put_Line ("Usage: cat ir1.json ir2.json | ir_merge_funcs");
      Put_Line ("  Reads all JSON arrays from stdin and merges elements into one array");
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Version then
      Put_Line ("ir_merge_funcs " & Version);
      Set_Exit_Status (Exit_Success); return;
   end if;
   if Show_Describe then
      Put_Line (Describe_Output);
      Set_Exit_Status (Exit_Success); return;
   end if;

   declare
      Input  : constant String := Read_Stdin;
      Merged : constant String := Merge_Arrays (Input);
   begin
      Put_Line (Merged);
      Set_Exit_Status (Exit_Success);
   end;

exception
   when others =>
      Put_Line (Standard_Error, "ERROR: Processing failed");
      Set_Exit_Status (Exit_Error);
end IR_Merge_Funcs;
