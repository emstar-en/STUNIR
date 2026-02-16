--  STUNIR Report Formatter Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Report_Formatter is

   procedure Copy_String
     (Source  : String;
      Target  : out String;
      Tgt_Len : out Natural)
   with Pre => Target'Length >= Source'Length,
        Post => Tgt_Len = Source'Length
   is
   begin
      Tgt_Len := Source'Length;
      for I in 1 .. Source'Length loop
         pragma Loop_Invariant (I <= Source'Length);
         Target (Target'First + I - 1) := Source (Source'First + I - 1);
      end loop;
      for I in Source'Length + 1 .. Target'Length loop
         Target (Target'First + I - 1) := ' ';
      end loop;
   end Copy_String;

   procedure Append_To_Buffer
     (Buffer   : in Out Output_Buffer;
      Position : in Out Output_Length;
      Text     : String;
      Status   : out Report_Status)
   is
   begin
      if Position + Text'Length > Max_Output_Length then
         Status := Buffer_Overflow;
         return;
      end if;
      for I in Text'Range loop
         pragma Loop_Invariant (Position + (I - Text'First) < Max_Output_Length);
         Buffer (Position + 1 + (I - Text'First)) := Text (I);
      end loop;
      Position := Position + Text'Length;
      Status := Success;
   end Append_To_Buffer;

   procedure Int_To_Str
     (Value  : Integer;
      Buffer : out String;
      Length : out Natural)
   with Pre => Buffer'Length >= 12
   is
      Temp   : Integer := abs Value;
      Buf    : String (1 .. 12) := (others => '0');
      Pos    : Natural := 12;
      Is_Neg : constant Boolean := Value < 0;
   begin
      if Temp = 0 then
         Buffer (Buffer'First) := '0';
         Length := 1;
         for I in Buffer'First + 1 .. Buffer'Last loop
            Buffer (I) := ' ';
         end loop;
         return;
      end if;
      while Temp > 0 and Pos > 0 loop
         pragma Loop_Invariant (Pos >= 1 and Pos <= 12);
         Buf (Pos) := Character'Val (48 + (Temp mod 10));
         Temp := Temp / 10;
         Pos := Pos - 1;
      end loop;
      Length := 12 - Pos;
      if Is_Neg then
         Length := Length + 1;
      end if;
      declare
         Out_Pos : Natural := Buffer'First;
      begin
         if Is_Neg then
            Buffer (Out_Pos) := '-';
            Out_Pos := Out_Pos + 1;
         end if;
         for I in Pos + 1 .. 12 loop
            pragma Loop_Invariant (Out_Pos <= Buffer'Last);
            if Out_Pos <= Buffer'Last then
               Buffer (Out_Pos) := Buf (I);
               Out_Pos := Out_Pos + 1;
            end if;
         end loop;
         for I in Out_Pos .. Buffer'Last loop
            Buffer (I) := ' ';
         end loop;
      end;
   end Int_To_Str;

   procedure Initialize_Report
     (Data   : out Report_Data;
      Schema : String;
      Title  : String;
      Format : Report_Format;
      Status : out Report_Status)
   is
      S_Len, T_Len : Natural;
   begin
      Data := Null_Report_Data;
      Copy_String (Schema, Data.Schema, S_Len);
      Data.Schema_Len := Schema_Length (S_Len);
      Copy_String (Title, Data.Title, T_Len);
      Data.Title_Len := Title_Length (T_Len);
      Data.Format := Format;
      Data.Is_Valid := True;
      Status := Success;
   end Initialize_Report;

   procedure Add_String_Item
     (Data   : in Out Report_Data;
      Key    : String;
      Value  : String;
      Status : out Report_Status)
   is
      New_Item : Report_Item := Null_Report_Item;
      K_Len, V_Len : Natural;
   begin
      Copy_String (Key, New_Item.Key, K_Len);
      New_Item.Key_Len := Key_Length (K_Len);
      Copy_String (Value, New_Item.Value, V_Len);
      New_Item.Value_Len := Value_Length (V_Len);
      New_Item.Kind := String_Value;
      New_Item.Is_Valid := True;
      Data.Flat_Count := Data.Flat_Count + 1;
      Data.Flat_Items (Data.Flat_Count) := New_Item;
      Status := Success;
   end Add_String_Item;

   procedure Add_Integer_Item
     (Data   : in Out Report_Data;
      Key    : String;
      Value  : Integer;
      Status : out Report_Status)
   is
      New_Item : Report_Item := Null_Report_Item;
      K_Len    : Natural;
      V_Buf    : String (1 .. 12);
      V_Len    : Natural;
   begin
      Copy_String (Key, New_Item.Key, K_Len);
      New_Item.Key_Len := Key_Length (K_Len);
      Int_To_Str (Value, V_Buf, V_Len);
      Copy_String (V_Buf (1 .. V_Len), New_Item.Value, V_Len);
      New_Item.Value_Len := Value_Length (V_Len);
      New_Item.Kind := Integer_Value;
      New_Item.Int_Val := Value;
      New_Item.Is_Valid := True;
      Data.Flat_Count := Data.Flat_Count + 1;
      Data.Flat_Items (Data.Flat_Count) := New_Item;
      Status := Success;
   end Add_Integer_Item;

   procedure Add_Boolean_Item
     (Data   : in Out Report_Data;
      Key    : String;
      Value  : Boolean;
      Status : out Report_Status)
   is
      New_Item : Report_Item := Null_Report_Item;
      K_Len    : Natural;
      Val_Str  : constant String := (if Value then "true" else "false");
      V_Len    : Natural;
   begin
      Copy_String (Key, New_Item.Key, K_Len);
      New_Item.Key_Len := Key_Length (K_Len);
      Copy_String (Val_Str, New_Item.Value, V_Len);
      New_Item.Value_Len := Value_Length (V_Len);
      New_Item.Kind := Boolean_Value;
      New_Item.Bool_Val := Value;
      New_Item.Is_Valid := True;
      Data.Flat_Count := Data.Flat_Count + 1;
      Data.Flat_Items (Data.Flat_Count) := New_Item;
      Status := Success;
   end Add_Boolean_Item;

   procedure Add_Section
     (Data         : in Out Report_Data;
      Section_Name : String;
      Status       : out Report_Status)
   is
      New_Sec : Report_Section := Null_Report_Section;
      N_Len   : Natural;
   begin
      Copy_String (Section_Name, New_Sec.Name, N_Len);
      New_Sec.Name_Len := Section_Name_Length (N_Len);
      New_Sec.Is_Valid := True;
      Data.Section_Total := Data.Section_Total + 1;
      Data.Sections (Data.Section_Total) := New_Sec;
      Data.Has_Sections := True;
      Status := Success;
   end Add_Section;

   procedure Generate_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   is
   begin
      case Data.Format is
         when Text_Format => Generate_Text_Report (Data, Output, Length, Status);
         when JSON_Format => Generate_JSON_Report (Data, Output, Length, Status);
         when HTML_Format => Generate_HTML_Report (Data, Output, Length, Status);
         when XML_Format  => Generate_XML_Report (Data, Output, Length, Status);
      end case;
   end Generate_Report;

   procedure Generate_Text_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   is
      Pos : Output_Length := 0;
      LF  : constant String := (1 => ASCII.LF);
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos, "=== ", Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, Data.Title (1 .. Data.Title_Len), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, " ===" & LF, Status);
      if Status /= Success then return; end if;
      for I in 1 .. Data.Flat_Count loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            E : Report_Item renames Data.Flat_Items (I);
         begin
            Append_To_Buffer (Output, Pos, E.Key (1 .. E.Key_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, ": ", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, LF, Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Length := Pos;
   end Generate_Text_Report;

   procedure Generate_JSON_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   is
      Pos : Output_Length := 0;
      Sorted : Item_Array := Data.Flat_Items;
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Sort_Items_By_Key (Sorted, Data.Flat_Count);
      Append_To_Buffer (Output, Pos, "{""schema"":""", Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, Data.Schema (1 .. Data.Schema_Len), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, """", Status);
      if Status /= Success then return; end if;
      for I in 1 .. Data.Flat_Count loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            E : Report_Item renames Sorted (I);
         begin
            Append_To_Buffer (Output, Pos, ",""", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Key (1 .. E.Key_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, """:", Status);
            if Status /= Success then return; end if;
            case E.Kind is
               when String_Value =>
                  Append_To_Buffer (Output, Pos, """", Status);
                  if Status /= Success then return; end if;
                  Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
                  if Status /= Success then return; end if;
                  Append_To_Buffer (Output, Pos, """", Status);
               when Integer_Value | Float_Value =>
                  Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
               when Boolean_Value =>
                  Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
               when Null_Value =>
                  Append_To_Buffer (Output, Pos, "null", Status);
            end case;
            if Status /= Success then return; end if;
         end;
      end loop;
      Append_To_Buffer (Output, Pos, "}", Status);
      Length := Pos;
   end Generate_JSON_Report;

   procedure Generate_HTML_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   is
      Pos : Output_Length := 0;
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos,
         "<!DOCTYPE html><html><head><title>", Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, Data.Title (1 .. Data.Title_Len), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos,
         "</title></head><body><h1>", Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, Data.Title (1 .. Data.Title_Len), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, "</h1><table>", Status);
      if Status /= Success then return; end if;
      for I in 1 .. Data.Flat_Count loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            E : Report_Item renames Data.Flat_Items (I);
         begin
            Append_To_Buffer (Output, Pos, "<tr><td>", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Key (1 .. E.Key_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, "</td><td>", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, "</td></tr>", Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Append_To_Buffer (Output, Pos, "</table></body></html>", Status);
      Length := Pos;
   end Generate_HTML_Report;

   procedure Generate_XML_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   is
      Pos : Output_Length := 0;
      LF  : constant String := (1 => ASCII.LF);
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos,
         "<?xml version=""1.0"" encoding=""UTF-8""?>" & LF, Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, "<report schema=""", Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, Data.Schema (1 .. Data.Schema_Len), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, """>" & LF, Status);
      if Status /= Success then return; end if;
      for I in 1 .. Data.Flat_Count loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            E : Report_Item renames Data.Flat_Items (I);
         begin
            Append_To_Buffer (Output, Pos, "  <item key=""", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Key (1 .. E.Key_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, """>", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, E.Value (1 .. E.Value_Len), Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, "</item>" & LF, Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Append_To_Buffer (Output, Pos, "</report>" & LF, Status);
      Length := Pos;
   end Generate_XML_Report;

   procedure Clear_Report (Data : in Out Report_Data) is
   begin
      Data.Flat_Items := (others => Null_Report_Item);
      Data.Flat_Count := 0;
      Data.Sections := (others => Null_Report_Section);
      Data.Section_Total := 0;
      Data.Has_Sections := False;
   end Clear_Report;

   function Total_Item_Count (Data : Report_Data) return Natural is
      Total : Natural := Data.Flat_Count;
   begin
      for S in 1 .. Data.Section_Total loop
         pragma Loop_Invariant (Total <= Natural (Max_Report_Items) +
                                (S - 1) * Natural (Max_Report_Items));
         Total := Total + Data.Sections (S).Item_Total;
      end loop;
      return Total;
   end Total_Item_Count;

   procedure Sort_Items_By_Key
     (Items : in Out Item_Array;
      Count : Item_Count)
   is
      Temp    : Report_Item;
      Swapped : Boolean;
   begin
      if Count <= 1 then
         return;
      end if;
      for I in 1 .. Count - 1 loop
         pragma Loop_Invariant (I <= Count - 1);
         Swapped := False;
         for J in 1 .. Count - I loop
            pragma Loop_Invariant (J <= Count - I);
            if Items (J).Key > Items (J + 1).Key then
               Temp := Items (J);
               Items (J) := Items (J + 1);
               Items (J + 1) := Temp;
               Swapped := True;
            end if;
         end loop;
         exit when not Swapped;
      end loop;
   end Sort_Items_By_Key;

end Report_Formatter;
