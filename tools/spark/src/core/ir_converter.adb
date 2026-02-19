--  STUNIR IR Converter Package Body
--  Converts spec.json to ir.json format
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;

with Ada.Strings.Fixed;
with Ada.Text_IO;
use Ada.Text_IO;

package body IR_Converter is

   use Ada.Strings;
   use Ada.Strings.Fixed;

   --  =======================================================================
   --  Internal Helper Functions
   --  =======================================================================

   function Parse_Spec_Version (Version_Str : String) return Natural is
      Result : Natural := 0;
   begin
      for I in Version_Str'Range loop
         if Version_Str (I) in '0' .. '9' then
            Result := Result * 10 +
                      (Character'Pos (Version_Str (I)) - Character'Pos ('0'));
         elsif Version_Str (I) = '.' then
            exit;  --  Only parse major version
         end if;
      end loop;
      return Result;
   end Parse_Spec_Version;

   procedure Parse_Parameter_Array
     (Parser     : in out Parser_State;
      Param_List :    out Parameter_List;
      Status     :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length,
      Post => Param_List.Count <= Max_Parameters
   is
      Param_Idx : Parameter_Index := 0;
   begin
      Param_List := (Count => 0, Params => (others => (Name => Identifier_Strings.Null_Bounded_String, Param_Type => Type_Name_Strings.Null_Bounded_String)));
      Status := Success;

      --  Expect array start
      if Current_Token (Parser) /= Token_Array_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Status);
      if Status /= Success then
         return;
      end if;

      --  Parse array elements
      while Current_Token (Parser) /= Token_Array_End and Status = Success loop
         if Param_Idx >= Max_Parameters then
            Status := Error_Too_Large;
            return;
         end if;

         Param_Idx := Param_Idx + 1;

         --  Expect object start for parameter
         if Current_Token (Parser) /= Token_Object_Start then
            Status := Error_Parse;
            return;
         end if;

         Next_Token (Parser, Status);
         if Status /= Success then
            return;
         end if;

         --  Parse parameter fields
         declare
            Param_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
            Param_Type_Str : Type_Name_String := Type_Name_Strings.Null_Bounded_String;
            Has_Name : Boolean := False;
            Has_Type : Boolean := False;
         begin
            while Current_Token (Parser) /= Token_Object_End and Status = Success loop
               declare
                  Member_Name  : Identifier_String;
                  Member_Value : JSON_String;
               begin
                  Parse_String_Member (Parser, Member_Name, Member_Value, Status);
                  if Status /= Success then
                     return;
                  end if;

                  declare
                     Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
                  begin
                     if Name_Str = "name" then
                        declare
                           Val_Str : constant String := JSON_Strings.To_String (Member_Value);
                        begin
                           if Val_Str'Length <= Max_Identifier_Length then
                              Param_Name := Identifier_Strings.To_Bounded_String (Val_Str);
                              Has_Name := True;
                           else
                              Status := Error_Too_Large;
                              return;
                           end if;
                        end;
                     elsif Name_Str = "type" then
                        declare
                           Val_Str : constant String := JSON_Strings.To_String (Member_Value);
                        begin
                           if Val_Str'Length <= Max_Type_Name_Length then
                              Param_Type_Str := Type_Name_Strings.To_Bounded_String (Val_Str);
                              Has_Type := True;
                           else
                              Status := Error_Too_Large;
                              return;
                           end if;
                        end;
                     end if;
                  end;

                  --  Check for comma or end of object
                  if Current_Token (Parser) = Token_Comma then
                     Next_Token (Parser, Status);
                  end if;
               end;
            end loop;

            if not Has_Name or not Has_Type then
               Status := Error_Parse;
               return;
            end if;

            Param_List.Params (Param_Idx) := (Name => Param_Name, Param_Type => Param_Type_Str);
         end;

         --  Expect object end
         if Current_Token (Parser) /= Token_Object_End then
            Status := Error_Parse;
            return;
         end if;

         Next_Token (Parser, Status);
         if Status /= Success then
            return;
         end if;

         --  Check for comma or end of array
         if Current_Token (Parser) = Token_Comma then
            Next_Token (Parser, Status);
         end if;
      end loop;

      --  Expect array end
      if Current_Token (Parser) /= Token_Array_End then
         Status := Error_Parse;
         return;
      end if;

      Param_List.Count := Param_Idx;

      --  Advance past array end
      Next_Token (Parser, Status);
   end Parse_Parameter_Array;

   procedure Parse_Function_Array
     (Parser  : in out Parser_State;
      Funcs   :    out Function_Collection;
      Status  :    out Status_Code)
   with
      Pre  => Parser.Position <= Max_JSON_Length,
      Post => Funcs.Count <= Max_Functions
   is
      Func_Idx : Function_Index := 0;
   begin
      Funcs := (Count => 0, Functions => (others => (Name => Identifier_Strings.Null_Bounded_String, Return_Type => Type_Name_Strings.Null_Bounded_String, Parameters => (Count => 0, Params => (others => (Name => Identifier_Strings.Null_Bounded_String, Param_Type => Type_Name_Strings.Null_Bounded_String))))));
      Status := Success;

      --  Expect array start
      if Current_Token (Parser) /= Token_Array_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Status);
      if Status /= Success then
         return;
      end if;

      --  Parse array elements
      while Current_Token (Parser) /= Token_Array_End and Status = Success loop
         if Func_Idx >= Max_Functions then
            Status := Error_Too_Large;
            return;
         end if;

         Func_Idx := Func_Idx + 1;

         --  Expect object start for function
         if Current_Token (Parser) /= Token_Object_Start then
            Status := Error_Parse;
            return;
         end if;

         Next_Token (Parser, Status);
         if Status /= Success then
            return;
         end if;

         --  Parse function fields
         declare
            Func_Name : Identifier_String := Identifier_Strings.Null_Bounded_String;
            Ret_Type  : Type_Name_String := Type_Name_Strings.Null_Bounded_String;
            Params    : Parameter_List := (Count => 0, Params => (others => (Name => Identifier_Strings.Null_Bounded_String, Param_Type => Type_Name_Strings.Null_Bounded_String)));
            Has_Name  : Boolean := False;
         begin
            while Current_Token (Parser) /= Token_Object_End and Status = Success loop
               declare
                  Member_Name  : Identifier_String;
                  Member_Value : JSON_String;
               begin
                  Parse_String_Member (Parser, Member_Name, Member_Value, Status);
                  if Status /= Success then
                     return;
                  end if;

                  declare
                     Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
                  begin
                     if Name_Str = "name" then
                        declare
                           Val_Str : constant String := JSON_Strings.To_String (Member_Value);
                        begin
                           if Val_Str'Length <= Max_Identifier_Length then
                              Func_Name := Identifier_Strings.To_Bounded_String (Val_Str);
                              Has_Name := True;
                           else
                              Status := Error_Too_Large;
                              return;
                           end if;
                        end;
                     elsif Name_Str = "return_type" then
                        declare
                           Val_Str : constant String := JSON_Strings.To_String (Member_Value);
                        begin
                           if Val_Str'Length <= Max_Type_Name_Length then
                              Ret_Type := Type_Name_Strings.To_Bounded_String (Val_Str);
                           else
                              Status := Error_Too_Large;
                              return;
                           end if;
                        end;
                     elsif Name_Str = "parameters" then
                        Parse_Parameter_Array (Parser, Params, Status);
                        if Status /= Success then
                           return;
                        end if;
                     end if;
                  end;

                  --  Check for comma or end of object
                  if Current_Token (Parser) = Token_Comma then
                     Next_Token (Parser, Status);
                  end if;
               end;
            end loop;

            if not Has_Name then
               Status := Error_Parse;
               return;
            end if;

            Funcs.Functions (Func_Idx) := (Name => Func_Name, Return_Type => Ret_Type, Parameters => Params);
         end;

         --  Expect object end
         if Current_Token (Parser) /= Token_Object_End then
            Status := Error_Parse;
            return;
         end if;

         Next_Token (Parser, Status);
         if Status /= Success then
            return;
         end if;

         --  Check for comma or end of array
         if Current_Token (Parser) = Token_Comma then
            Next_Token (Parser, Status);
         end if;
      end loop;

      --  Expect array end
      if Current_Token (Parser) /= Token_Array_End then
         Status := Error_Parse;
         return;
      end if;

      Funcs.Count := Func_Idx;

      --  Advance past array end
      Next_Token (Parser, Status);
   end Parse_Function_Array;

   --  =======================================================================
   --  Public Operations
   --  =======================================================================

   procedure Parse_Spec_JSON
     (JSON_Content : in     JSON_String;
      Spec         :    out Spec_Input_Data;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
   begin
      Spec := (Schema_Version => Identifier_Strings.Null_Bounded_String,
               Module => (Name => Identifier_Strings.Null_Bounded_String,
                         Functions => (Count => 0, Functions => (others => (Name => Identifier_Strings.Null_Bounded_String, Return_Type => Type_Name_Strings.Null_Bounded_String, Parameters => (Count => 0, Params => (others => (Name => Identifier_Strings.Null_Bounded_String, Param_Type => Type_Name_Strings.Null_Bounded_String))))))));
      Status := Success;

      Initialize_Parser (Parser, JSON_Content, Status);
      if Status /= Success then
         return;
      end if;

      --  Expect object start
      Next_Token (Parser, Status);
      if Status /= Success or else Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Status);
      if Status /= Success then
         return;
      end if;

      --  Parse root object members
      while Current_Token (Parser) /= Token_Object_End and Status = Success loop
         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Status);
            if Status /= Success then
               return;
            end if;

            declare
               Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
            begin
               if Name_Str = "schema_version" then
                  declare
                     Val_Str : constant String := JSON_Strings.To_String (Member_Value);
                  begin
                     if Val_Str'Length <= Max_Identifier_Length then
                        Spec.Schema_Version := Identifier_Strings.To_Bounded_String (Val_Str);
                     else
                        Status := Error_Too_Large;
                        return;
                     end if;
                  end;
               elsif Name_Str = "module" then
                  --  Parse module object
                  if Current_Token (Parser) /= Token_Object_Start then
                     Status := Error_Parse;
                     return;
                  end if;

                  Next_Token (Parser, Status);
                  if Status /= Success then
                     return;
                  end if;

                  while Current_Token (Parser) /= Token_Object_End and Status = Success loop
                     declare
                        Mod_Member_Name  : Identifier_String;
                        Mod_Member_Value : JSON_String;
                     begin
                        Parse_String_Member (Parser, Mod_Member_Name, Mod_Member_Value, Status);
                        if Status /= Success then
                           return;
                        end if;

                        declare
                           Mod_Name_Str : constant String := Identifier_Strings.To_String (Mod_Member_Name);
                        begin
                           if Mod_Name_Str = "name" then
                              declare
                                 Val_Str : constant String := JSON_Strings.To_String (Mod_Member_Value);
                              begin
                                 if Val_Str'Length <= Max_Identifier_Length then
                                    Spec.Module.Name := Identifier_Strings.To_Bounded_String (Val_Str);
                                 else
                                    Status := Error_Too_Large;
                                    return;
                                 end if;
                              end;
                           elsif Mod_Name_Str = "functions" then
                              Parse_Function_Array (Parser, Spec.Module.Functions, Status);
                              if Status /= Success then
                                 return;
                              end if;
                           end if;
                        end;

                        --  Check for comma or end of module object
                        if Current_Token (Parser) = Token_Comma then
                           Next_Token (Parser, Status);
                        end if;
                     end;
                  end loop;

                  --  Expect module object end
                  if Current_Token (Parser) /= Token_Object_End then
                     Status := Error_Parse;
                     return;
                  end if;

                  Next_Token (Parser, Status);
                  if Status /= Success then
                     return;
                  end if;
               end if;
            end;

            --  Check for comma or end of root object
            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Status);
            end if;
         end;
      end loop;

      --  Expect object end
      if Current_Token (Parser) /= Token_Object_End then
         Status := Error_Parse;
         return;
      end if;

      Status := Success;
   end Parse_Spec_JSON;

   procedure Validate_Spec
     (Spec   : in     Spec_Input_Data;
      Status :    out Status_Code)
   is
   begin
      Status := Success;

      --  Check schema version
      if Identifier_Strings.Length (Spec.Schema_Version) = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      --  Check module name
      if Identifier_Strings.Length (Spec.Module.Name) = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      --  Check functions
      if Spec.Module.Functions.Count = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      --  Validate each function
      for I in 1 .. Spec.Module.Functions.Count loop
         declare
            Func : constant Function_Signature := Spec.Module.Functions.Functions (I);
         begin
            if Identifier_Strings.Length (Func.Name) = 0 then
               Status := Error_Invalid_Input;
               return;
            end if;

            --  Validate parameters
            for J in 1 .. Func.Parameters.Count loop
               if Identifier_Strings.Length (Func.Parameters.Params (J).Name) = 0 then
                  Status := Error_Invalid_Input;
                  return;
               end if;
            end loop;
         end;
      end loop;
   end Validate_Spec;

   procedure Convert_Spec_To_IR
     (Spec        : in     Spec_Input_Data;
      Module_Name : in     Identifier_String;
      IR          :    out IR_Module;
      Status      :    out Status_Code)
   is
   begin
      IR := (Schema_Version => Spec.Schema_Version,
             IR_Version => Identifier_Strings.To_Bounded_String ("1.0"),
             Module_Name => Module_Name,
             Functions => (Count => 0, Functions => (others => (Name => Identifier_Strings.Null_Bounded_String, Return_Type => Type_Name_Strings.Null_Bounded_String, Parameters => (Count => 0, Params => (others => (Name => Identifier_Strings.Null_Bounded_String, Param_Type => Type_Name_Strings.Null_Bounded_String))), Steps => (Count => 0, Steps => (others => (Step_Type => Step_Noop, Target => Identifier_Strings.Null_Bounded_String, Source => Identifier_Strings.Null_Bounded_String, Value => Identifier_Strings.Null_Bounded_String)))))));
      Status := Success;

      --  Copy functions with IR structure
      IR.Functions.Count := Spec.Module.Functions.Count;

      for I in 1 .. Spec.Module.Functions.Count loop
         declare
            Spec_Func : constant Function_Signature := Spec.Module.Functions.Functions (I);
            IR_Func   : IR_Function;
         begin
            IR_Func.Name := Spec_Func.Name;
            IR_Func.Return_Type := Spec_Func.Return_Type;
            IR_Func.Parameters := Spec_Func.Parameters;

            --  Add a single NOOP step (placeholder for actual body)
            IR_Func.Steps.Count := 1;
            IR_Func.Steps.Steps (1) := (Step_Type => Step_Noop,
                                        Target => Identifier_Strings.Null_Bounded_String,
                                        Source => Identifier_Strings.Null_Bounded_String,
                                        Value => Identifier_Strings.To_Bounded_String ("placeholder"));

            IR.Functions.Functions (I) := IR_Func;
         end;
      end loop;
   end Convert_Spec_To_IR;

   procedure Generate_IR_JSON
     (IR          : in     IR_Module;
      JSON_Output :    out JSON_String;
      Status      :    out Status_Code)
   is
      Output : String (1 .. Max_JSON_Length);
      Pos    : Positive := 1;

      procedure Append (S : String) with
         Pre  => Pos + S'Length - 1 <= Max_JSON_Length,
         Post => Pos = Pos'Old + S'Length
      is
      begin
         Output (Pos .. Pos + S'Length - 1) := S;
         Pos := Pos + S'Length;
      end Append;

      procedure Append_Identifier (Id : Identifier_String) is
         Str : constant String := Identifier_Strings.To_String (Id);
      begin
         Append ("""");
         Append (Str);
         Append ("""");
      end Append_Identifier;

      procedure Append_Type_Name (T : Type_Name_String) is
         Str : constant String := Type_Name_Strings.To_String (T);
      begin
         Append ("""");
         Append (Str);
         Append ("""");
      end Append_Type_Name;

      procedure Append_Parameter (Param : Parameter; Is_Last : Boolean) is
      begin
         Append ("{""name"":");
         Append_Identifier (Param.Name);
         Append (",""type"":");
         Append_Type_Name (Param.Param_Type);
         Append ("}");
         if not Is_Last then
            Append (",");
         end if;
      end Append_Parameter;

      procedure Append_Step (Step : IR_Step; Is_Last : Boolean) is
      begin
         Append ("{""step_type"":");
         case Step.Step_Type is
            when Step_Noop =>
               Append ("""noop""");
            when Step_Assign =>
               Append ("""assign""");
            when Step_Call =>
               Append ("""call""");
            when Step_Return =>
               Append ("""return""");
         end case;
         Append (",""target"":");
         Append_Identifier (Step.Target);
         Append (",""source"":");
         Append_Identifier (Step.Source);
         Append (",""value"":");
         Append_Identifier (Step.Value);
         Append ("}");
         if not Is_Last then
            Append (",");
         end if;
      end Append_Step;

      procedure Append_IR_Function (IR_Func : IR_Function; Is_Last : Boolean) is
      begin
         Append ("{""name"":");
         Append_Identifier (IR_Func.Name);
         Append (",""return_type"":");
         Append_Type_Name (IR_Func.Return_Type);

         --  Parameters
         Append (",""parameters"":[");
         for I in 1 .. IR_Func.Parameters.Count loop
            Append_Parameter (IR_Func.Parameters.Params (I), I = IR_Func.Parameters.Count);
         end loop;
         Append ("],""steps"":[");

         --  Steps
         for I in 1 .. IR_Func.Steps.Count loop
            Append_Step (IR_Func.Steps.Steps (I), I = IR_Func.Steps.Count);
         end loop;
         Append ("]}");

         if not Is_Last then
            Append (",");
         end if;
      end Append_IR_Function;

   begin
      --  Build JSON output
      Append ("{""schema_version"":");
      Append_Identifier (IR.Schema_Version);
      Append (",""ir_version"":");
      Append_Identifier (IR.IR_Version);
      Append (",""module_name"":");
      Append_Identifier (IR.Module_Name);
      Append (",""functions"":[");

      for I in 1 .. IR.Functions.Count loop
         Append_IR_Function (IR.Functions.Functions (I), I = IR.Functions.Count);
      end loop;

      Append ("]}");

      if Pos > Max_JSON_Length then
         Status := Error_Too_Large;
         return;
      end if;

      JSON_Output := JSON_Strings.To_Bounded_String (Output (1 .. Pos - 1));
      Status := Success;
   end Generate_IR_JSON;

   procedure Convert_Argument
     (Spec_Arg : in     Parameter;
      IR_Arg   :    out Parameter;
      Status   :    out Status_Code)
   is
   begin
      IR_Arg := Spec_Arg;
      Status := Success;
   end Convert_Argument;

   procedure Convert_Function
     (Spec_Func : in     Function_Signature;
      IR_Func   :    out IR_Function;
      Status    :    out Status_Code)
   is
   begin
      IR_Func.Name := Spec_Func.Name;
      IR_Func.Return_Type := Spec_Func.Return_Type;
      IR_Func.Parameters := Spec_Func.Parameters;
      IR_Func.Steps := (Count => 0, Steps => (others => (Step_Type => Step_Noop, Target => Identifier_Strings.Null_Bounded_String, Source => Identifier_Strings.Null_Bounded_String, Value => Identifier_Strings.Null_Bounded_String)));
      Status := Success;
   end Convert_Function;

   procedure Add_Noop_Step
     (IR_Func : in out IR_Function;
      Status  :    out Status_Code)
   is
   begin
      if IR_Func.Steps.Count >= Max_Steps then
         Status := Error_Too_Large;
         return;
      end if;

      IR_Func.Steps.Count := IR_Func.Steps.Count + 1;
      IR_Func.Steps.Steps (IR_Func.Steps.Count) := (Step_Type => Step_Noop,
                                                     Target => Identifier_Strings.Null_Bounded_String,
                                                     Source => Identifier_Strings.Null_Bounded_String,
                                                     Value => Identifier_Strings.To_Bounded_String ("noop"));
      Status := Success;
   end Add_Noop_Step;

   procedure Process_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Status      :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK

      Input_File  : File_Type;
      Output_File : File_Type;
      File_Content : String (1 .. Max_JSON_Length);
      Content_Len  : Natural := 0;
      JSON_Content : JSON_String;
      Spec         : Spec_Input_Data;
      IR           : IR_Module;
      JSON_Output  : JSON_String;
   begin
      Status := Success;

      --  Read input file
      begin
         Open (Input_File, In_File, Path_Strings.To_String (Input_Path));
         while not End_Of_File (Input_File) and Content_Len < Max_JSON_Length loop
            declare
               Line : constant String := Get_Line (Input_File);
               New_Len : constant Natural := Content_Len + Line'Length;
            begin
               if New_Len <= Max_JSON_Length then
                  File_Content (Content_Len + 1 .. New_Len) := Line;
                  Content_Len := New_Len;
               else
                  Close (Input_File);
                  Status := Error_File_IO;
                  return;
               end if;
            end;
         end loop;
         Close (Input_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      if Content_Len = 0 then
         Status := Error_Invalid_Input;
         return;
      end if;

      JSON_Content := JSON_Strings.To_Bounded_String (File_Content (1 .. Content_Len));

      --  Parse spec JSON
      Parse_Spec_JSON (JSON_Content, Spec, Status);
      if Status /= Success then
         return;
      end if;

      --  Validate spec
      Validate_Spec (Spec, Status);
      if Status /= Success then
         return;
      end if;

      --  Convert to IR
      Convert_Spec_To_IR (Spec, Module_Name, IR, Status);
      if Status /= Success then
         return;
      end if;

      --  Generate IR JSON
      Generate_IR_JSON (IR, JSON_Output, Status);
      if Status /= Success then
         return;
      end if;

      --  Write output file
      begin
         Create (Output_File, Out_File, Path_Strings.To_String (Output_Path));
         Put (Output_File, JSON_Strings.To_String (JSON_Output));
         Close (Output_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;

      Status := Success;
   end Process_Spec_File;

end IR_Converter;