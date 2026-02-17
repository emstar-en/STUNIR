with Ada.Command_Line; with Ada.Text_IO; with Ada.Strings.Unbounded;

package body Module_To_IR is

   procedure Process_Module is
      Metadata : constant String := Get_Input_From_Stdin;
      IR_Output : Unbounded_String;
   begin
      -- Start building the IR JSON output
      IR_Output := To_Unbounded_String("{\n  \"schema\": \"stunir_flat_ir_v1\",\n  \"ir_version\": \"0.1.0\",\n  \"module_name\": \"") & 
                  Get_Json_Value(Metadata, "name") & 
                  To_Unbounded_String("\",\n  \"description\": \"") & 
                  Get_Json_Value(Metadata, "description") & 
                  To_Unbounded_String("\"\n}"));

      Ada.Text_IO.Put_Line(To_String(IR_Output));
   exception
      when others =>
         Ada.Text_IO.Put_Line("ERROR: " & Ada.Exceptions.Exception_Information);
         Set_Exit_Failure;
   end Process_Module;

end Module_To_IR;