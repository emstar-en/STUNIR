--  STUNIR_JSON_Parser - JSON parsing utilities implementation
--  SPARK-compliant implementation

pragma SPARK_Mode (On);

with Ada.Characters.Handling;

package body STUNIR_JSON_Parser is

   use Ada.Characters.Handling;
   use STUNIR_Types;

   --  Helper functions
   function Is_Whitespace (C : Character) return Boolean is
   begin
      return C = ' ' or C = ASCII.HT or C = ASCII.LF or C = ASCII.CR;
   end Is_Whitespace;

   function Is_Digit (C : Character) return Boolean is
   begin
      return C in '0' .. '9';
   end Is_Digit;

   procedure Skip_Whitespace (State : in out Parser_State) is
      Pos : Natural := State.Position;
   begin
      while Pos <= JSON_Strings.Length (State.Input) loop
         declare
            C : constant Character := JSON_Strings.Element (State.Input, Pos);
         begin
            exit when not Is_Whitespace (C);
            if C = ASCII.LF then
               State.Line := State.Line + 1;
               State.Column := 1;
            else
               State.Column := State.Column + 1;
            end if;
            Pos := Pos + 1;
         end;
      end loop;
      State.Position := Pos;
   end Skip_Whitespace;

   procedure Parse_String (State : in out Parser_State; Status : out Status_Code) is
      Pos : Natural := State.Position + 1;  --  Skip opening quote
      Result : JSON_String := JSON_Strings.Null_Bounded_String;
      Result_Len : Natural := 0;
   begin
      while Pos <= JSON_Strings.Length (State.Input) loop
         declare
            C : constant Character := JSON_Strings.Element (State.Input, Pos);
         begin
            if C = '"' then
               State.Position := Pos + 1;
               State.Column := State.Column + (Pos - State.Position) + 1;
               State.Current_Token := Token_String;
               State.Token_Value := Result;
               Status := Success;
               return;
            elsif C = '\' then
               --  Handle escape sequences
               Pos := Pos + 1;
               if Pos <= JSON_Strings.Length (State.Input) then
                  declare
                     Next_C : constant Character := JSON_Strings.Element (State.Input, Pos);
                     Escape_Char : Character;
                  begin
                     case Next_C is
                        when '"' => Escape_Char := '"';
                        when '\' => Escape_Char := '\';
                        when '/' => Escape_Char := '/';
                        when 'b' => Escape_Char := ASCII.BS;
                        when 'f' => Escape_Char := ASCII.FF;
                        when 'n' => Escape_Char := ASCII.LF;
                        when 'r' => Escape_Char := ASCII.CR;
                        when 't' => Escape_Char := ASCII.HT;
                        when others =>
                           Status := Error;
                           return;
                     end case;
                     if Result_Len < JSON_Strings.Max_Length (Result) then
                        Result := JSON_Strings.Insert (Result, Result_Len + 1, Escape_Char);
                        Result_Len := Result_Len + 1;
                     else
                        Status := Error;
                        return;
                     end if;
                  end;
               else
                  Status := Error;
                  return;
               end if;
            elsif C < ' ' then
               --  Control characters not allowed in strings
               Status := Error;
               return;
            else
               if Result_Len < JSON_Strings.Max_Length (Result) then
                  Result := JSON_Strings.Insert (Result, Result_Len + 1, C);
                  Result_Len := Result_Len + 1;
               else
                  Status := Error;
                  return;
               end if;
            end if;
            Pos := Pos + 1;
         end;
      end loop;
      Status := Error;  --  Unterminated string
   end Parse_String;

   procedure Parse_Number (State : in out Parser_State; Status : out Status_Code) is
      Pos : Natural := State.Position;
      Has_Digits : Boolean := False;
      Num_Str : JSON_String;
      Num_Len : Natural;
   begin
      --  Optional minus sign
      if Pos <= JSON_Strings.Length (State.Input) and then
         JSON_Strings.Element (State.Input, Pos) = '-'
      then
         Pos := Pos + 1;
      end if;

      --  Integer part
      while Pos <= JSON_Strings.Length (State.Input) and then
            Is_Digit (JSON_Strings.Element (State.Input, Pos))
      loop
         Has_Digits := True;
         Pos := Pos + 1;
      end loop;

      --  Fractional part
      if Pos <= JSON_Strings.Length (State.Input) and then
         JSON_Strings.Element (State.Input, Pos) = '.'
      then
         Pos := Pos + 1;
         while Pos <= JSON_Strings.Length (State.Input) and then
               Is_Digit (JSON_Strings.Element (State.Input, Pos))
         loop
            Has_Digits := True;
            Pos := Pos + 1;
         end loop;
      end if;

      --  Exponent part
      if Pos <= JSON_Strings.Length (State.Input) and then
         (JSON_Strings.Element (State.Input, Pos) = 'e' or
          JSON_Strings.Element (State.Input, Pos) = 'E')
      then
         Pos := Pos + 1;
         if Pos <= JSON_Strings.Length (State.Input) and then
            (JSON_Strings.Element (State.Input, Pos) = '+' or
             JSON_Strings.Element (State.Input, Pos) = '-')
         then
            Pos := Pos + 1;
         end if;
         while Pos <= JSON_Strings.Length (State.Input) and then
               Is_Digit (JSON_Strings.Element (State.Input, Pos))
         loop
            Has_Digits := True;
            Pos := Pos + 1;
         end loop;
      end if;

      if not Has_Digits then
         Status := Error;
         return;
      end if;

      --  Extract the number string
      Num_Len := Pos - State.Position;
      if Num_Len > 0 and Num_Len <= JSON_Strings.Max_Length (Num_Str) then
         Num_Str := JSON_Strings.Slice (State.Input, State.Position, Pos - 1);
         State.Token_Value := Num_Str;
      else
         Status := Error;
         return;
      end if;

      State.Column := State.Column + (Pos - State.Position);
      State.Position := Pos;
      State.Current_Token := Token_Number;
      Status := Success;
   end Parse_Number;

   procedure Parse_Keyword (State : in out Parser_State; Status : out Status_Code) is
      Pos : Natural := State.Position;
      C : Character;
      Keyword_Value : JSON_String;
   begin
      if Pos > JSON_Strings.Length (State.Input) then
         Status := Error;
         return;
      end if;

      C := JSON_Strings.Element (State.Input, Pos);

      case C is
         when 't' =>  --  true
            if Pos + 3 <= JSON_Strings.Length (State.Input) and then
               JSON_Strings.Slice (State.Input, Pos, Pos + 3) = "true"
            then
               State.Current_Token := Token_True;
               Keyword_Value := JSON_Strings.To_Bounded_String ("true");
               State.Token_Value := Keyword_Value;
               State.Position := Pos + 4;
               State.Column := State.Column + 4;
               Status := Success;
            else
               Status := Error;
            end if;

         when 'f' =>  --  false
            if Pos + 4 <= JSON_Strings.Length (State.Input) and then
               JSON_Strings.Slice (State.Input, Pos, Pos + 4) = "false"
            then
               State.Current_Token := Token_False;
               Keyword_Value := JSON_Strings.To_Bounded_String ("false");
               State.Token_Value := Keyword_Value;
               State.Position := Pos + 5;
               State.Column := State.Column + 5;
               Status := Success;
            else
               Status := Error;
            end if;

         when 'n' =>  --  null
            if Pos + 3 <= JSON_Strings.Length (State.Input) and then
               JSON_Strings.Slice (State.Input, Pos, Pos + 3) = "null"
            then
               State.Current_Token := Token_Null;
               Keyword_Value := JSON_Strings.To_Bounded_String ("null");
               State.Token_Value := Keyword_Value;
               State.Position := Pos + 4;
               State.Column := State.Column + 4;
               Status := Success;
            else
               Status := Error;
            end if;

         when others =>
            Status := Error;
      end case;
   end Parse_Keyword;

   --  Public procedures

   procedure Initialize_Parser
     (State  : in out Parser_State;
      Input  : JSON_String;
      Status : out Status_Code)
   is
   begin
      State.Input := Input;
      State.Position := 1;
      State.Line := 1;
      State.Column := 1;
      State.Current_Token := Token_EOF;
      State.Token_Value := JSON_Strings.Null_Bounded_String;
      Status := Success;
   end Initialize_Parser;

   procedure Next_Token
     (State  : in out Parser_State;
      Status : out Status_Code)
   is
      C : Character;
      Token_Value : JSON_String;
   begin
      Skip_Whitespace (State);

      if State.Position > JSON_Strings.Length (State.Input) then
         State.Current_Token := Token_EOF;
         Status := EOF_Reached;
         return;
      end if;

      C := JSON_Strings.Element (State.Input, State.Position);

      case C is
         when '{' =>
            State.Current_Token := Token_LBrace;
            Token_Value := JSON_Strings.To_Bounded_String ("{");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when '}' =>
            State.Current_Token := Token_RBrace;
            Token_Value := JSON_Strings.To_Bounded_String ("}");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when '[' =>
            State.Current_Token := Token_LBracket;
            Token_Value := JSON_Strings.To_Bounded_String ("[");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when ']' =>
            State.Current_Token := Token_RBracket;
            Token_Value := JSON_Strings.To_Bounded_String ("]");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when ':' =>
            State.Current_Token := Token_Colon;
            Token_Value := JSON_Strings.To_Bounded_String (":");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when ',' =>
            State.Current_Token := Token_Comma;
            Token_Value := JSON_Strings.To_Bounded_String (",");
            State.Token_Value := Token_Value;
            State.Position := State.Position + 1;
            State.Column := State.Column + 1;
            Status := Success;

         when '"' =>
            Parse_String (State, Status);

         when '0' .. '9' | '-' =>
            Parse_Number (State, Status);

         when 't' | 'f' | 'n' =>
            Parse_Keyword (State, Status);

         when others =>
            State.Current_Token := Token_Error;
            Status := Error;
      end case;
   end Next_Token;

   function Validate_JSON (Input : String) return Boolean is
      State : Parser_State;
      Status : Status_Code;
      Input_Bounded : JSON_String;
      Depth : Natural := 0;
      Max_Depth : constant := 1000;  --  Prevent stack overflow
   begin
      if Input'Length = 0 then
         return False;
      end if;

      if Input'Length > Max_JSON_Length then
         return False;
      end if;

      begin
         Input_Bounded := JSON_Strings.To_Bounded_String (Input);
      exception
         when others =>
            return False;
      end;

      Initialize_Parser (State, Input_Bounded, Status);
      if Status /= Success then
         return False;
      end if;

      --  Simple validation: just check that tokens are valid
      --  A full parser would validate structure here
      loop
         Next_Token (State, Status);
         exit when Status = EOF_Reached;
         if Status = Error then
            return False;
         end if;
         if State.Current_Token = Token_LBrace or
            State.Current_Token = Token_LBracket then
            Depth := Depth + 1;
            if Depth > Max_Depth then
               return False;  --  Too deeply nested
            end if;
         elsif State.Current_Token = Token_RBrace or
               State.Current_Token = Token_RBracket then
            if Depth = 0 then
               return False;  --  Unmatched closing bracket
            end if;
            Depth := Depth - 1;
         end if;
      end loop;

      return Depth = 0;  --  All brackets matched
   end Validate_JSON;

   procedure Expect_Token
     (State    : in out Parser_State;
      Expected : Token_Type;
      Status   : out Status_Code)
   is
   begin
      Next_Token (State, Status);
      if Status /= Success then
         return;
      end if;

      if State.Current_Token /= Expected then
         Status := Error;
      end if;
   end Expect_Token;

end STUNIR_JSON_Parser;