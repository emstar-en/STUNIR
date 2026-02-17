with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

package json_read is

   -- JSON validation state tracking
   type json_state is (start, in_object, in_array, in_string, in_value);
   current_state : json_state := start;
   brace_count : Integer := 0;
   bracket_count : Integer := 0;
   string_depth : Integer := 0;

   procedure validate_json(input : String);

end json_read;

with Ada.Exceptions;

package body json_read is

   -- Helper procedures for JSON validation
   procedure process_char(c : Character) is
      use Ada.Strings.Unbounded;
      to_unbounded : Unbounded_String;
   begin
      case current_state is
         when start =>
            if c = '{' then
               current_state := in_object;
               brace_count := 1;
            elsif c = '[' then
               current_state := in_array;
               bracket_count := 1;
            else
               raise JSON_Validation_Error with "Unexpected character at start: " & c;
            end if;

         when in_object | in_array =>
            case c is
               when '{' => brace_count := brace_count + 1;
               when '}' =>
                  if brace_count > 1 then
                     brace_count := brace_count - 1;
                  else
                     raise JSON_Validation_Error with "Unexpected closing brace";
                  end if;
               when '[' => bracket_count := bracket_count + 1;
               when ']' =>
                  if bracket_count > 1 then
                     bracket_count := bracket_count - 1;
                  else
                     raise JSON_Validation_Error with "Unexpected closing bracket";
                  end if;
               when '"' =>
                  string_depth := string_depth + 1;
               when others => null; -- Skip whitespace and other characters
            end case;

         when in_string =>
            if c = '"' then
               string_depth := string_depth - 1;
               if string_depth = 0 then
                  current_state := in_value;
               end if;
            end if;

         when in_value => null; -- Skip values between structures

      end case;
   exception
      when others =>
         raise JSON_Validation_Error with "Invalid JSON structure";
   end process_char;

   procedure validate_json(input : String) is
      use Ada.Strings.Unbounded;
      to_unbounded : Unbounded_String := To_Unbounded_String(input);
      c : Character;
   begin
      for i in 1 .. Length(to_unbounded) loop
         c := Element(to_unbounded, i);
         process_char(c);
      end loop;

      if brace_count /= 0 or bracket_count /= 0 then
         raise JSON_Validation_Error with "Unbalanced braces/brackets";
      elsif string_depth > 0 then
         raise JSON_Validation_Error with "Unclosed string";
      end if;
   exception
      when others =>
         raise JSON_Validation_Error with Ada.Exceptions.Exception_Information;
   end validate_json;

begin
   -- Main processing
   declare
      use Ada.Text_IO;
      input_file : File_Type;
      output_file : File_Type;
      input_path : Unbounded_String;
      json_content : String := (1 .. 0 => '<');
      valid : Boolean := False;
   begin
      if Command_Line.Argument_Count = 0 then
         -- Read from stdin
         validate_json(Standard_Input);
         Put_Line(Standard_Output, Standard_Input);
      else
         -- Read from file
         input_path := To_Unbounded_String(Command_Line.Argument(1));
         Open(Input_File, In_File, To_String(input_path));

         declare
            buffer : String := (1 .. 4096 => '<');
            length : Integer;
         begin
            loop
               Get_Line(Input_File, buffer);
               validate_json(buffer);
               Put_Line(Standard_Output, buffer);
            exception
               when End_Error =>
                  Close(Input_File);
                  valid := True;
                  exit;
               when others =>
                  Close(Input_File);
                  raise JSON_Validation_Error with Ada.Exceptions.Exception_Information;
            end loop;
         end;
      end if;

      -- Exit codes
      if not valid then
         Set_Exit_Status(1); -- Validation error
      else
         Set_Exit_Status(0); -- Success
      end if;
   exception
      when others =>
         Set_Exit_Status(1);
         raise;
   end;

exception
   when JSON_Validation_Error =>
      Set_Exit_Status(1);
end json_read;