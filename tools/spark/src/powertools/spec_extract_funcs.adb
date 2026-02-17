with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;

package Spec_Extract_Funcs is

   -- Main procedure to extract functions from spec JSON
   procedure Extract_Functions;

private

   -- Helper function to check if a string contains valid JSON array pattern
   function Is_JSON_Array (json_str : String) return Boolean;

end Spec_Extract_Funcs;

package body Spec_Extract_Funcs is

   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   procedure Extract_Functions is
      spec_json : Unbounded_String;
      functions_array : Unbounded_String := To_Unbounded_String("[]");
      found_functions : Boolean := False;
      start_pos, end_pos : Integer;
      temp_str : Unbounded_String;

   begin
      -- Read entire input from stdin
      while not End_Of_Line (Standard_Input) loop
         spec_json := spec_json & To_Unbounded_String(Get_Line(Standard_Input));
      end loop;

      -- Check if input is valid JSON array pattern for functions
      if Is_JSON_Array (To_String(spec_json)) then
         -- Find the "functions": [...] portion
         start_pos := Index (To_String(spec_json), """functions""");
         if start_pos > 0 then
            -- Extract from "functions" to matching closing bracket
            temp_str := To_Unbounded_String(To_String(spec_json));
            declare
               depth : Integer := 1;
               pos : Integer := start_pos + 9; -- Skip past "functions"
            begin
               while pos <= Length(temp_str) and then depth > 0 loop
                  if Slice (temp_str, pos, 1) = """ then
                     depth := depth - 1;
                  elsif Slice (temp_str, pos, 1) = "[" then
                     depth := depth + 1;
                  end if;
                  pos := pos + 1;
               end loop;

               -- Extract the functions array portion
               if depth <= 0 and pos > start_pos then
                  functions_array := Slice (temp_str, start_pos + 9, pos - (start_pos + 9) - 1);
                  found_functions := True;
               end if;
            end;
         end if;
      end if;

      -- Output the extracted functions array or empty array
      Put_Line (To_String(functions_array));

      -- Set exit status based on success
      if not Is_JSON_Array (To_String(spec_json)) then
         Set_Exit_Status (1); -- Invalid spec JSON
      end if;
   exception
      when others =>
         Put_Line (Standard_Error, "Error: Failed to process input");
         Set_Exit_Status (2);
   end Extract_Functions;

   function Is_JSON_Array (json_str : String) return Boolean is
      pos : Integer := 1;
   begin
      -- Check for opening bracket
      if json_str'Length < 3 then
         return False;
      end if;

      -- Skip whitespace and check for [
      while pos <= json_str'Length and then json_str(pos) = ' ' loop
         pos := pos + 1;
      end loop;
      if pos > json_str'Length or then json_str(pos) /= '[' then
         return False;
      end if;

      -- Check closing bracket at the end (with possible whitespace)
      while json_str'Length >= pos and then json_str(json_str'Length - 1) = ' ' loop
         json_str := json_str(1..json_str'Length - 2);
      end loop;
      return json_str(json_str'Last) = ']';
   exception
      when others =>
         return False;
   end Is_JSON_Array;

begin
   Extract_Functions;
end Spec_Extract_Funcs;