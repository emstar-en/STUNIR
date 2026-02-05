--  STUNIR JSON Parser Package Body
--  Streaming JSON parser implementation
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Ada.Characters.Handling;
with Ada.Strings.Fixed;

package body STUNIR_JSON_Parser is

   use Ada.Characters.Handling;
   use Ada.Strings;
   use Ada.Strings.Fixed;

   --  =======================================================================
   --  Internal Helper Functions
   --  =======================================================================

   function Is_Whitespace (C : Character) return Boolean is
     (C = ' ' or C = ASCII.HT or C = ASCII.LF or C = ASCII.CR);

   function Is_Digit (C : Character) return Boolean is
     (C in '0' .. '9');

   function Is_Hex_Digit (C : Character) return Boolean is
     (Is_Digit (C) or (C in 'a' .. 'f') or (C in 'A' .. 'F'));

   procedure Skip_Whitespace
     (State : in out Parser_State)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => State.Position >= State.Position'Old
   is
      Current_Pos : Positive := State.Position;
      Input_Str   : constant String := JSON_Strings.To_String (State.Input);
   begin
      while Current_Pos <= Input_Str'Length
         and then Is_Whitespace (Input_Str (Current_Pos))
      loop
         if Input_Str (Current_Pos) = ASCII.LF then
            State.Line := State.Line + 1;
            State.Column := 1;
         else
            State.Column := State.Column + 1;
         end if;
         Current_Pos := Current_Pos + 1;
      end loop;
      State.Position := Current_Pos;
   end Skip_Whitespace;

   procedure Parse_String_Literal
     (State  : in out Parser_State;
      Result :    out JSON_String;
      Status :    out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => State.Position >= State.Position'Old
   is
      Input_Str   : constant String := JSON_Strings.To_String (State.Input);
      Start_Pos   : constant Positive := State.Position;
      Current_Pos : Positive := State.Position;
      Temp_Str    : String (1 .. Max_Identifier_Length);
      Temp_Len    : Natural := 0;
   begin
      Result := JSON_Strings.Null_Bounded_String;
      Status := Error_Parse;

      if Current_Pos > Input_Str'Length then
         return;
      end if;

      --  Check for opening quote
      if Input_Str (Current_Pos) /= '"' then
         return;
      end if;

      Current_Pos := Current_Pos + 1;
      State.Column := State.Column + 1;

      --  Parse string content
      while Current_Pos <= Input_Str'Length loop
         declare
            C : constant Character := Input_Str (Current_Pos);
         begin
            exit when C = '"';  --  End of string

            if C = '\' then
               --  Escape sequence
               if Current_Pos + 1 > Input_Str'Length then
                  return;  --  Invalid escape at end
               end if;

               Current_Pos := Current_Pos + 1;
               State.Column := State.Column + 1;

               declare
                  Esc_C : constant Character := Input_Str (Current_Pos);
               begin
                  if Temp_Len < Max_Identifier_Length then
                     case Esc_C is
                        when '"' | '\' | '/' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := Esc_C;
                        when 'b' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := ASCII.BS;
                        when 'f' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := ASCII.FF;
                        when 'n' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := ASCII.LF;
                        when 'r' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := ASCII.CR;
                        when 't' =>
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := ASCII.HT;
                        when 'u' =>
                           --  Unicode escape \uXXXX - simplified handling
                           if Current_Pos + 4 > Input_Str'Length then
                              return;
                           end if;
                           --  Just skip the hex digits for now
                           Temp_Len := Temp_Len + 1;
                           Temp_Str (Temp_Len) := '?';  --  Placeholder
                           Current_Pos := Current_Pos + 4;
                           State.Column := State.Column + 4;
                        when others =>
                           return;  --  Invalid escape
                     end case;
                  end if;
               end;
            elsif C < ' ' then
               --  Control character in string (invalid)
               return;
            else
               if Temp_Len < Max_Identifier_Length then
                  Temp_Len := Temp_Len + 1;
                  Temp_Str (Temp_Len) := C;
               end if;
            end if;

            Current_Pos := Current_Pos + 1;
            State.Column := State.Column + 1;
         end;
      end loop;

      --  Check for closing quote
      if Current_Pos > Input_Str'Length or else Input_Str (Current_Pos) /= '"' then
         return;
      end if;

      Current_Pos := Current_Pos + 1;
      State.Column := State.Column + 1;
      State.Position := Current_Pos;

      Result := JSON_Strings.To_Bounded_String (Temp_Str (1 .. Temp_Len));
      Status := Success;
   end Parse_String_Literal;

   procedure Parse_Number
     (State  : in out Parser_State;
      Result :    out JSON_String;
      Status :    out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => State.Position >= State.Position'Old
   is
      Input_Str   : constant String := JSON_Strings.To_String (State.Input);
      Start_Pos   : constant Positive := State.Position;
      Current_Pos : Positive := State.Position;
   begin
      Result := JSON_Strings.Null_Bounded_String;
      Status := Error_Parse;

      if Current_Pos > Input_Str'Length then
         return;
      end if;

      --  Optional minus sign
      if Input_Str (Current_Pos) = '-' then
         Current_Pos := Current_Pos + 1;
         State.Column := State.Column + 1;
      end if;

      --  Integer part
      if Current_Pos <= Input_Str'Length
         and then Input_Str (Current_Pos) = '0'
      then
         Current_Pos := Current_Pos + 1;
         State.Column := State.Column + 1;
      elsif Current_Pos <= Input_Str'Length
         and then Is_Digit (Input_Str (Current_Pos))
      then
         while Current_Pos <= Input_Str'Length
            and then Is_Digit (Input_Str (Current_Pos))
         loop
            Current_Pos := Current_Pos + 1;
            State.Column := State.Column + 1;
         end loop;
      else
         return;  --  Not a valid number
      end if;

      --  Fractional part
      if Current_Pos <= Input_Str'Length
         and then Input_Str (Current_Pos) = '.'
      then
         Current_Pos := Current_Pos + 1;
         State.Column := State.Column + 1;

         if Current_Pos > Input_Str'Length
            or else not Is_Digit (Input_Str (Current_Pos))
         then
            return;  --  Expected digit after decimal point
         end if;

         while Current_Pos <= Input_Str'Length
            and then Is_Digit (Input_Str (Current_Pos))
         loop
            Current_Pos := Current_Pos + 1;
            State.Column := State.Column + 1;
         end loop;
      end if;

      --  Exponent part
      if Current_Pos <= Input_Str'Length
         and then (Input_Str (Current_Pos) = 'e' or Input_Str (Current_Pos) = 'E')
      then
         Current_Pos := Current_Pos + 1;
         State.Column := State.Column + 1;

         --  Optional sign
         if Current_Pos <= Input_Str'Length
            and then (Input_Str (Current_Pos) = '+' or Input_Str (Current_Pos) = '-')
         then
            Current_Pos := Current_Pos + 1;
            State.Column := State.Column + 1;
         end if;

         if Current_Pos > Input_Str'Length
            or else not Is_Digit (Input_Str (Current_Pos))
         then
            return;  --  Expected digit in exponent
         end if;

         while Current_Pos <= Input_Str'Length
            and then Is_Digit (Input_Str (Current_Pos))
         loop
            Current_Pos := Current_Pos + 1;
            State.Column := State.Column + 1;
         end loop;
      end if;

      --  Extract the number string
      declare
         Num_Str : constant String := Input_Str (Start_Pos .. Current_Pos - 1);
      begin
         if Num_Str'Length <= Max_Identifier_Length then
            Result := JSON_Strings.To_Bounded_String (Num_Str);
            Status := Success;
         else
            Status := Error_Too_Large;
            return;
         end if;
      end;

      State.Position := Current_Pos;
   end Parse_Number;

   --  =======================================================================
   --  Public Operations
   --  =======================================================================

   procedure Initialize_Parser
     (State  : out Parser_State;
      Input  : in  JSON_String;
      Status : out Status_Code)
   is
   begin
      if JSON_Strings.Length (Input) = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      State := Parser_State'(
         Input         => Input,
         Position      => 1,
         Line          => 1,
         Column        => 1,
         Nesting       => (others => Nest_Object),
         Nesting_Level => 0,
         Current_Token => Token_EOF,
         Token_Value   => JSON_Strings.Null_Bounded_String
      );

      Status := Success;
   end Initialize_Parser;

   procedure Next_Token
     (State  : in out Parser_State;
      Status : out Status_Code)
   is
      Input_Str : constant String := JSON_Strings.To_String (State.Input);
   begin
      Status := Success;

      --  Skip whitespace
      Skip_Whitespace (State);

      if State.Position > Input_Str'Length then
         State.Current_Token := Token_EOF;
         return;
      end if;

      declare
         C : constant Character := Input_Str (State.Position);
      begin
         case C is
            when '{' =>
               State.Current_Token := Token_Object_Start;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when '}' =>
               State.Current_Token := Token_Object_End;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when '[' =>
               State.Current_Token := Token_Array_Start;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when ']' =>
               State.Current_Token := Token_Array_End;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when ':' =>
               State.Current_Token := Token_Colon;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when ',' =>
               State.Current_Token := Token_Comma;
               State.Position := State.Position + 1;
               State.Column := State.Column + 1;

            when '"' =>
               Parse_String_Literal (State, State.Token_Value, Status);
               if Status = Success then
                  State.Current_Token := Token_String;
               end if;

            when 't' =>
               --  true
               if State.Position + 3 <= Input_Str'Length
                  and then Input_Str (State.Position .. State.Position + 3) = "true"
               then
                  State.Current_Token := Token_True;
                  State.Position := State.Position + 4;
                  State.Column := State.Column + 4;
               else
                  Status := Error_Parse;
               end if;

            when 'f' =>
               --  false
               if State.Position + 4 <= Input_Str'Length
                  and then Input_Str (State.Position .. State.Position + 4) = "false"
               then
                  State.Current_Token := Token_False;
                  State.Position := State.Position + 5;
                  State.Column := State.Column + 5;
               else
                  Status := Error_Parse;
               end if;

            when 'n' =>
               --  null
               if State.Position + 3 <= Input_Str'Length
                  and then Input_Str (State.Position .. State.Position + 3) = "null"
               then
                  State.Current_Token := Token_Null;
                  State.Position := State.Position + 4;
                  State.Column := State.Column + 4;
               else
                  Status := Error_Parse;
               end if;

            when '-' | '0' .. '9' =>
               Parse_Number (State, State.Token_Value, Status);
               if Status = Success then
                  State.Current_Token := Token_Number;
               end if;

            when others =>
               Status := Error_Parse;
         end case;
      end;
   end Next_Token;

   function Current_Token (State : Parser_State) return Token_Kind is
     (State.Current_Token);

   function Token_String_Value (State : Parser_State) return JSON_String is
     (State.Token_Value);

   procedure Expect_Token
     (State      : in out Parser_State;
      Expected   : in     Token_Kind;
      Status     :    out Status_Code)
   is
   begin
      Next_Token (State, Status);
      if Status = Success and then State.Current_Token /= Expected then
         Status := Error_Parse;
      end if;
   end Expect_Token;

   procedure Skip_Value
     (State  : in out Parser_State;
      Status : out Status_Code)
   is
      Depth : Natural := 0;
   begin
      Status := Success;

      case State.Current_Token is
         when Token_String | Token_Number | Token_True | Token_False | Token_Null =>
            --  Simple values - just advance to next token
            Next_Token (State, Status);

         when Token_Object_Start | Token_Array_Start =>
            --  Complex values - need to skip entire structure
            Depth := 1;
            while Depth > 0 and Status = Success loop
               Next_Token (State, Status);
               case State.Current_Token is
                  when Token_Object_Start | Token_Array_Start =>
                     Depth := Depth + 1;
                  when Token_Object_End | Token_Array_End =>
                     Depth := Depth - 1;
                  when Token_EOF =>
                     Status := Error_Parse;
                  when others =>
                     null;
               end case;
            end loop;

            if Status = Success then
               Next_Token (State, Status);
            end if;

         when others =>
            --  Already at end token or unexpected
            Next_Token (State, Status);
      end case;
   end Skip_Value;

   procedure Parse_String_Member
     (State       : in out Parser_State;
      Member_Name :    out Identifier_String;
      Member_Value:    out JSON_String;
      Status      :    out Status_Code)
   is
      Temp_Str : JSON_String;
   begin
      Member_Name := Identifier_Strings.Null_Bounded_String;
      Member_Value := JSON_Strings.Null_Bounded_String;
      Status := Error_Parse;

      --  Expect string token (member name)
      if State.Current_Token /= Token_String then
         return;
      end if;

      Temp_Str := State.Token_Value;

      --  Convert to identifier string
      declare
         Str_Content : constant String := JSON_Strings.To_String (Temp_Str);
      begin
         if Str_Content'Length > Max_Identifier_Length then
            Status := Error_Too_Large;
            return;
         end if;
         Member_Name := Identifier_Strings.To_Bounded_String (Str_Content);
      end;

      --  Expect colon
      Next_Token (State, Status);
      if Status /= Success or else State.Current_Token /= Token_Colon then
         Status := Error_Parse;
         return;
      end if;

      --  Get value
      Next_Token (State, Status);
      if Status /= Success then
         return;
      end if;

      case State.Current_Token is
         when Token_String =>
            Member_Value := State.Token_Value;
            Status := Success;

         when Token_Number | Token_True | Token_False | Token_Null =>
            --  For non-string values, store the literal text
            declare
               Start_Pos : constant Positive := State.Position;
            begin
               --  Re-parse to get the raw text
               Skip_Whitespace (State);
               --  Token value already contains the text for these types
               Member_Value := State.Token_Value;
               Status := Success;
            end;

         when others =>
            Status := Error_Parse;
      end case;
   end Parse_String_Member;

   function Is_At_End (State : Parser_State) return Boolean is
     (State.Current_Token = Token_EOF);

   function Get_Position_Line (State : Parser_State) return Positive is
     (State.Line);

   function Get_Position_Column (State : Parser_State) return Positive is
     (State.Column);

end STUNIR_JSON_Parser;