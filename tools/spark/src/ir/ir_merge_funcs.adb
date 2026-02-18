with Ada.Command_Line; with Ada.Text_IO; with Ada.Strings.Unbounded;

package body IR_Merge_Funcs is

   procedure Process_Functions is
      Input_Data : constant String := Get_Input_From_Stdin;
      Functions  : Unbounded_String;
      First       : Boolean := True;
   begin
      -- Start building the functions array
      Functions := To_Unbounded_String("[")

      -- Process each function JSON object
      declare
         Line : String;
      begin
         while Get_Line(Line) loop
            if not First then
               Functions := Append(Functions, To_Unbounded_String(","));
            end if;
            First := False;

            -- Add the function JSON to our array
            Functions := Append(Functions, To_Unbounded_String(Line));
         end loop;
      end;

      -- Complete the IR JSON structure
      Functions := Append(Functions, To_Unbounded_String("]"));

      Ada.Text_IO.Put_Line(To_String(Functions));
   exception
      when others =>
         Ada.Text_IO.Put_Line("ERROR: " & Ada.Exceptions.Exception_Information);
         Set_Exit_Failure;
   end Process_Functions;

end IR_Merge_Funcs;