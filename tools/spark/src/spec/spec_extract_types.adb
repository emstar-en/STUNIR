with Ada.Command_Line;
with Ada.Text_IO;
with Ada.Strings.Unbounded;
with Ada.Strings.Maps;

package body Spec_Extract_Types is

   -- Local types and constants
   type Json_Parse_Error is new Exception;
   type Json_Validation_Error is new Exception;

   -- JSON parsing utilities (simplified for this example)
   function Is_Json_Array (Text : String) return Boolean;
   function Extract_Json_Array (Text : String; Array_Name : String) return Unbounded_String;

   -- Main processing
   procedure Process_Spec is
      Spec_Json : constant String := Get_Input_From_Stdin;
      Types_Array : Unbounded_String;
   begin
      if not Is_Valid_Json (Spec_Json) then
         Ada.Text_IO.Put_Line ("INVALID JSON");
         raise Json_Parse_Error;
      end if;

      Types_Array := Extract_Json_Array (Spec_Json, "module.types");

      -- Output the types array
      Put_Line (To_String (Types_Array));

   exception
      when Json_Parse_Error =>
         Ada.Text_IO.Put_Line ("INVALID JSON");
         Set_Exit_Failure;
      when others =>
         Ada.Text_IO.Put_Line ("ERROR: " & Ada.Exceptions.Exception_Information (Ada.Exceptions.Current_Exception));
         Set_Exit_Failure;
   end Process_Spec;

   -- Helper functions
   function Is_Valid_Json (Text : String) return Boolean is
      -- Simplified JSON validation - in real implementation would use proper parser
      Valid_Braces : constant Character := '{';
      Count : Integer := 0;
   begin
      for C of Text loop
         if C = Valid_Braces then
            Count := Count + 1;
         elsif C = '}' then
            Count := Count - 1;
         end if;

         -- Check for balanced braces and basic structure
         if Count < 0 or (Count > 0 and Text(Text'Last) /= '}') then
            return False;
         end if;
      end loop;
      return Count = 0;
   end Is_Valid_Json;

   function Extract_Json_Array (Text : String; Array_Name : String) return Unbounded_String is
      -- Find the array in JSON format
      -- This is a simplified version - real implementation would use proper JSON parsing
      Pos : constant Integer := Text'Find (Array_Name & ": [");
   begin
      if Pos = 0 then
         return To_Unbounded_String ("[]"); -- Return empty array if not found
      end if;

      -- Extract the content between [ and ]
      declare
         Start_Pos : constant Integer := Pos + Array_Name'Length + 3;
         End_Pos : integer := Text'Last - 1;
      begin
         while End_Pos > Start_Pos and then Text (End_Pos) /= ']' loop
            End_Pos := End_Pos - 1;
         end loop;

         if End_Pos < Start_Pos then
            return To_Unbounded_String ("[]");
         else
            return To_Unbounded_String (Text (Start_Pos .. End_Pos));
         end if;
      end;
   end Extract_Json_Array;

end Spec_Extract_Types;