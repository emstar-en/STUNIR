with Ada.Command_Line; with Ada.Text_IO; with Ada.Strings.Unbounded;

package body Func_To_IR is

   function Normalize_Type (Input_Type : String) return String is
      Lower_Type : constant String := To_Lower(Input_Type);
   begin
      if Lower_Type = "int" then
         return "i32";
      elsif Lower_Type = "uint" then
         return "u32";
      elsif Lower_Type = "long" then
         return "i64";
      elsif Lower_Type = "short" then
         return "i16";
      elsif Lower_Type = "byte" then
         return "i8";
      elsif Lower_Type = "char" then
         return "u8";
      elsif Lower_Type = "float" then
         return "f32";
      elsif Lower_Type = "double" then
         return "f64";
      elsif Lower_Type = "bool" then
         return "bool";
      elsif Lower_Type = "string" then
         return "str";
      else
         return Input_Type;
      end if;
   end Normalize_Type;

   procedure Process_Function is
      Function_Spec : constant String := Get_Input_From_Stdin;
      IR_Output     : Unbounded_String;
   begin
      -- Start building the IR JSON output
      IR_Output := To_Unbounded_String("{\n  \"name\": \"") & 
                  Get_Json_Value(Function_Spec, "name") & 
                  To_Unbounded_String("\",\n  \"return_type\": \"") & 
                  Normalize_Type(Get_Json_Value(Function_Spec, "returns")) & 
                  To_Unbounded_String("\",\n  \"args\": [");

      -- Process each parameter
      declare
         Params : constant String := Get_Json_Value(Function_Spec, "params");
         Param_Count : Integer := 0;
      begin
         for I in 1..Get_Array_Length(Params) loop
            if I > 1 then
               IR_Output := Append(IR_Output, To_Unbounded_String(","));
            end if;

            declare
               Param_Name : constant String := Get_Json_Value(Params, Integer'Image(I), "name");
               Param_Type  : constant String := Normalize_Type(Get_Json_Value(Params, Integer'Image(I), "type"));
            begin
               IR_Output := Append(IR_Output, To_Unbounded_String("{\"name\": \"") & 
                                   To_Unbounded_String(Param_Name) & 
                                   To_Unbounded_String("\", \"type\": \"") & 
                                   To_Unbounded_String(Param_Type) & 
                                   To_Unbounded_String("\"}"));
            end;
         end loop;
      end;

      -- Complete the IR JSON structure
      IR_Output := Append(IR_Output, To_Unbounded_String("],\n  \"steps\": [],\n  \"is_public\": true\n}"));

      Ada.Text_IO.Put_Line(To_String(IR_Output));
   exception
      when others =>
         Ada.Text_IO.Put_Line("ERROR: " & Ada.Exceptions.Exception_Information);
         Set_Exit_Failure;
   end Process_Function;

end Func_To_IR;