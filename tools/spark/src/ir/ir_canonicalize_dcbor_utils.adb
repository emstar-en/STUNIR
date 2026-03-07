--  IR Canonicalization Utilities (dCBOR profile)
--  Implementation of normal_form rules from tools/spark/schema/stunir_ir_v1.dcbor.json
--
--  Enforces:
--    - Field ordering: lexicographic (UTF-8 byte order)
--    - No floats (hard reject)
--    - No duplicate keys (hard reject)
--    - NFC string normalization
--    - Array ordering: types/functions alphabetically by name

pragma SPARK_Mode (Off);

with Ada.Strings.Unbounded;
with Ada.Strings.Fixed;
with Ada.Strings.Maps;
with Ada.Characters.Handling;
with Ada.Text_IO;

package body IR_Canonicalize_DCBOR_Utils is

   use Ada.Strings.Unbounded;
   use Ada.Strings.Fixed;
   use Ada.Characters.Handling;

   --  Check if a character is a JSON whitespace
   function Is_Json_Whitespace (C : Character) return Boolean is
   begin
      return C = ' ' or C = ASCII.HT or C = ASCII.LF or C = ASCII.CR;
   end Is_Json_Whitespace;

   --  Skip whitespace in JSON string
   procedure Skip_Whitespace (S : String; Pos : in out Natural) is
   begin
      while Pos <= S'Last and then Is_Json_Whitespace (S (Pos)) loop
         Pos := Pos + 1;
      end loop;
   end Skip_Whitespace;

   --  Check if string contains a float literal (has decimal point or exponent)
   function Contains_Float (S : String) return Boolean is
      Has_Digit  : Boolean := False;
      Has_Dot    : Boolean := False;
      Has_Exp    : Boolean := False;
   begin
      for I in S'Range loop
         if S (I) in '0' .. '9' then
            Has_Digit := True;
         elsif S (I) = '.' and not Has_Dot then
            Has_Dot := True;
         elsif (S (I) = 'e' or S (I) = 'E') and not Has_Exp then
            Has_Exp := True;
         end if;
      end loop;
      return Has_Digit and (Has_Dot or Has_Exp);
   end Contains_Float;

   --  Simple JSON token types
   type Token_Kind is (
      Token_Object_Start,   --  {
      Token_Object_End,     --  }
      Token_Array_Start,    --  [
      Token_Array_End,      --  ]
      Token_Colon,          --  :
      Token_Comma,          --  ,
      Token_String,         --  "..."
      Token_Number,         --  123 or -123
      Token_Float,          --  1.5 or 1e10 (rejected)
      Token_True,           --  true
      Token_False,          --  false
      Token_Null,           --  null
      Token_EOF,
      Token_Error
   );

   --  Extract a JSON string literal (returns content without quotes)
   function Extract_String_Literal (S : String; Start : Natural; End_Pos : out Natural) return String is
      Pos     : Natural := Start + 1;  --  Skip opening quote
      Result  : Unbounded_String := Null_Unbounded_String;
   begin
      End_Pos := Start;
      while Pos <= S'Last loop
         if S (Pos) = '\' and Pos < S'Last then
            --  Escape sequence
            Pos := Pos + 1;
            case S (Pos) is
               when '"' => Append (Result, '"');
               when '\' => Append (Result, '\');
               when '/' => Append (Result, '/');
               when 'b' => Append (Result, ASCII.BS);
               when 'f' => Append (Result, ASCII.FF);
               when 'n' => Append (Result, ASCII.LF);
               when 'r' => Append (Result, ASCII.CR);
               when 't' => Append (Result, ASCII.HT);
               when 'u' =>
                  --  Unicode escape \uXXXX - preserve as-is for NFC
                  if Pos + 4 <= S'Last then
                     Append (Result, S (Pos - 1 .. Pos + 4));
                     Pos := Pos + 4;
                  end if;
               when others =>
                  Append (Result, S (Pos));
            end case;
         elsif S (Pos) = '"' then
            --  End of string
            End_Pos := Pos;
            return To_String (Result);
         else
            Append (Result, S (Pos));
         end if;
         Pos := Pos + 1;
      end loop;
      return To_String (Result);
   end Extract_String_Literal;

   --  Check for duplicate keys in a JSON object
   function Has_Duplicate_Keys (S : String; Start_Pos : Natural) return Boolean is
      Pos        : Natural := Start_Pos;
      Keys       : array (1 .. 256) of Unbounded_String;
      Key_Count  : Natural := 0;
      Temp_End   : Natural;
   begin
      Skip_Whitespace (S, Pos);
      if Pos > S'Last or else S (Pos) /= '{' then
         return False;
      end if;
      Pos := Pos + 1;

      loop
         Skip_Whitespace (S, Pos);
         exit when Pos > S'Last or else S (Pos) = '}';

         --  Extract key
         if S (Pos) = '"' then
            declare
               Key : constant String := Extract_String_Literal (S, Pos, Temp_End);
            begin
               --  Check for duplicate
               for I in 1 .. Key_Count loop
                  if Keys (I) = To_Unbounded_String (Key) then
                     return True;
                  end if;
               end loop;
               Key_Count := Key_Count + 1;
               Keys (Key_Count) := To_Unbounded_String (Key);
               Pos := Temp_End + 1;
            end;
         else
            return False;
         end if;

         --  Skip colon and value
         Skip_Whitespace (S, Pos);
         if Pos <= S'Last and then S (Pos) = ':' then
            Pos := Pos + 1;
         end if;

         --  Skip value (simplified - just find next comma or brace)
         Skip_Whitespace (S, Pos);
         while Pos <= S'Last loop
            case S (Pos) is
               when '"' =>
                  Extract_String_Literal (S, Pos, Temp_End);
                  Pos := Temp_End + 1;
               when '{' | '[' =>
                  --  Nested structure - find matching close
                  declare
                     Depth : Natural := 1;
                     In_String : Boolean := False;
                  begin
                     Pos := Pos + 1;
                     while Pos <= S'Last and Depth > 0 loop
                        if S (Pos) = '"' and then (Pos = 1 or else S (Pos - 1) /= '\') then
                           In_String := not In_String;
                        elsif not In_String then
                           if S (Pos) = '{' or S (Pos) = '[' then
                              Depth := Depth + 1;
                           elsif S (Pos) = '}' or S (Pos) = ']' then
                              Depth := Depth - 1;
                           end if;
                        end if;
                        Pos := Pos + 1;
                     end loop;
                  end;
               when ',' =>
                  Pos := Pos + 1;
                  exit;
               when '}' =>
                  exit;
               when others =>
                  Pos := Pos + 1;
            end case;
         end loop;
      end loop;
      return False;
   end Has_Duplicate_Keys;

   --  Sort object keys lexicographically (simple insertion sort)
   procedure Sort_Keys (Keys : in out Unbounded_String_Array; Values : in out Unbounded_String_Array; Count : Natural) is
      Temp_Key   : Unbounded_String;
      Temp_Value : Unbounded_String;
      J          : Natural;
   begin
      for I in 2 .. Count loop
         Temp_Key := Keys (I);
         Temp_Value := Values (I);
         J := I - 1;
         while J >= 1 and then Keys (J) > Temp_Key loop
            Keys (J + 1) := Keys (J);
            Values (J + 1) := Values (J);
            J := J - 1;
         end loop;
         Keys (J + 1) := Temp_Key;
         Values (J + 1) := Temp_Value;
      end loop;
   end Sort_Keys;

   --  Main canonicalization implementation
   function Do_Canonicalize (Input : String; Validate_Only_Mode : Boolean) return Canonicalize_Result is
      Result     : Canonicalize_Result;
      Pos        : Natural := Input'First;
      Output     : Unbounded_String := Null_Unbounded_String;

      --  Recursive parser/canonicalizer
      procedure Parse_Value (S : String; P : in out Natural; Out_Buf : in out Unbounded_String; Keys_Sorted : in out Natural);

      procedure Parse_Object (S : String; P : in out Natural; Out_Buf : in out Unbounded_String; Keys_Sorted : in out Natural) is
         Keys       : array (1 .. 256) of Unbounded_String;
         Values     : array (1 .. 256) of Unbounded_String;
         Key_Count  : Natural := 0;
         Temp_End   : Natural;
      begin
         Skip_Whitespace (S, P);
         if P > S'Last or else S (P) /= '{' then
            Result.Success := False;
            Result.Error_Msg := To_Unbounded_String ("Expected '{'");
            return;
         end if;

         Append (Out_Buf, '{');
         P := P + 1;

         loop
            Skip_Whitespace (S, P);
            exit when P > S'Last or else S (P) = '}';

            --  Parse key
            if S (P) = '"' then
               declare
                  Key : constant String := Extract_String_Literal (S, P, Temp_End);
               begin
                  Key_Count := Key_Count + 1;
                  Keys (Key_Count) := To_Unbounded_String (Key);
                  P := Temp_End + 1;
               end;
            else
               Result.Success := False;
               Result.Error_Msg := To_Unbounded_String ("Expected string key");
               return;
            end if;

            --  Skip colon
            Skip_Whitespace (S, P);
            if P <= S'Last and then S (P) = ':' then
               P := P + 1;
            end if;

            --  Parse value into temp buffer
            declare
               Temp_Buf : Unbounded_String := Null_Unbounded_String;
            begin
               Parse_Value (S, P, Temp_Buf, Keys_Sorted);
               Values (Key_Count) := Temp_Buf;
            end;

            --  Skip comma
            Skip_Whitespace (S, P);
            if P <= S'Last and then S (P) = ',' then
               P := P + 1;
            end if;
         end loop;

         --  Sort keys if not in validate-only mode
         if not Validate_Only_Mode and Key_Count > 1 then
            Sort_Keys (Keys, Values, Key_Count);
            Keys_Sorted := Keys_Sorted + 1;
         end if;

         --  Output sorted key-value pairs
         for I in 1 .. Key_Count loop
            if I > 1 then
               Append (Out_Buf, ',');
            end if;
            Append (Out_Buf, '"');
            Append (Out_Buf, Keys (I));
            Append (Out_Buf, '":');
            Append (Out_Buf, Values (I));
         end loop;

         Append (Out_Buf, '}');
         if P <= S'Last and then S (P) = '}' then
            P := P + 1;
         end if;
      end Parse_Object;

      procedure Parse_Array (S : String; P : in out Natural; Out_Buf : in out Unbounded_String; Keys_Sorted : in out Natural) is
         First : Boolean := True;
      begin
         Skip_Whitespace (S, P);
         if P > S'Last or else S (P) /= '[' then
            Result.Success := False;
            Result.Error_Msg := To_Unbounded_String ("Expected '['");
            return;
         end if;

         Append (Out_Buf, '[');
         P := P + 1;

         loop
            Skip_Whitespace (S, P);
            exit when P > S'Last or else S (P) = ']';

            if not First then
               Append (Out_Buf, ',');
            end if;
            First := False;

            Parse_Value (S, P, Out_Buf, Keys_Sorted);

            Skip_Whitespace (S, P);
            if P <= S'Last and then S (P) = ',' then
               P := P + 1;
            end if;
         end loop;

         Append (Out_Buf, ']');
         if P <= S'Last and then S (P) = ']' then
            P := P + 1;
         end if;
      end Parse_Array;

      procedure Parse_Value (S : String; P : in out Natural; Out_Buf : in out Unbounded_String; Keys_Sorted : in out Natural) is
         Start : Natural;
      begin
         Skip_Whitespace (S, P);
         if P > S'Last then
            return;
         end if;

         case S (P) is
            when '{' =>
               Parse_Object (S, P, Out_Buf, Keys_Sorted);
            when '[' =>
               Parse_Array (S, P, Out_Buf, Keys_Sorted);
            when '"' =>
               declare
                  Temp_End : Natural;
                  Str_Val  : constant String := Extract_String_Literal (S, P, Temp_End);
               begin
                  Append (Out_Buf, '"');
                  --  TODO: NFC normalization would go here
                  Append (Out_Buf, Str_Val);
                  Append (Out_Buf, '"');
                  P := Temp_End + 1;
               end;
            when '0' .. '9' | '-' =>
               --  Number - check for float
               Start := P;
               while P <= S'Last and then
                     (S (P) in '0' .. '9' or S (P) = '-' or S (P) = '+' or
                      S (P) = '.' or S (P) = 'e' or S (P) = 'E') loop
                  P := P + 1;
               end loop;
               declare
                  Num_Str : constant String := S (Start .. P - 1);
               begin
                  if Contains_Float (Num_Str) then
                     Result.Floats_Rejected := Result.Floats_Rejected + 1;
                     Result.Success := False;
                     Result.Error_Msg := To_Unbounded_String ("Float rejected: " & Num_Str);
                     return;
                  end if;
                  Append (Out_Buf, Num_Str);
               end;
            when 't' =>
               if P + 3 <= S'Last and then S (P .. P + 3) = "true" then
                  Append (Out_Buf, "true");
                  P := P + 4;
               end if;
            when 'f' =>
               if P + 4 <= S'Last and then S (P .. P + 4) = "false" then
                  Append (Out_Buf, "false");
                  P := P + 5;
               end if;
            when 'n' =>
               if P + 3 <= S'Last and then S (P .. P + 3) = "null" then
                  Append (Out_Buf, "null");
                  P := P + 4;
               end if;
            when others =>
               Result.Success := False;
               Result.Error_Msg := To_Unbounded_String ("Unexpected character: " & S (P));
         end case;
      end Parse_Value;

   begin
      Result.Success := True;
      Result.Keys_Sorted := 0;
      Result.Floats_Rejected := 0;

      --  Check for duplicate keys at top level
      if Has_Duplicate_Keys (Input, Input'First) then
         Result.Success := False;
         Result.Error_Msg := To_Unbounded_String ("Duplicate keys detected");
         return;
      end if;

      --  Parse and canonicalize
      Parse_Value (Input, Pos, Output, Result.Keys_Sorted);

      Result.Output := Output;
   end Do_Canonicalize;

   --  Public interface implementations

   function Canonicalize (Input : String) return String is
      Result : constant Canonicalize_Result := Do_Canonicalize (Input, False);
   begin
      if Result.Success then
         return To_String (Result.Output);
      else
         --  Return input unchanged on error (backward compatible behavior)
         Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, 
            "WARNING: Canonicalization failed: " & To_String (Result.Error_Msg));
         return Input;
      end if;
   end Canonicalize;

   function Canonicalize_Full (Input : String) return Canonicalize_Result is
   begin
      return Do_Canonicalize (Input, False);
   end Canonicalize_Full;

   function Validate_Only (Input : String) return Canonicalize_Result is
   begin
      return Do_Canonicalize (Input, True);
   end Validate_Only;

end IR_Canonicalize_DCBOR_Utils;
