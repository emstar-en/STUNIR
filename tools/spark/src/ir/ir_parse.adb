--  IR Parse Micro-Tool Body
--  Parses IR JSON into internal representation
--  Phase: 2 (IR)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;
with Ada.Text_IO;
use Ada.Text_IO;

package body IR_Parse is

   --  Internal helper to initialize IR_Data with defaults
   procedure Init_IR_Data (IR : out IR_Data) is
   begin
      IR.Schema_Version  := Identifier_Strings.Null_Bounded_String;
      IR.IR_Version      := Identifier_Strings.Null_Bounded_String;
      IR.Module_Name     := Identifier_Strings.Null_Bounded_String;
      
      --  Initialize imports
      IR.Imports.Count := 0;
      for I in Import_Index range 1 .. Max_Imports loop
         IR.Imports.Imports (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Imports.Imports (I).From_Module := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize exports
      IR.Exports.Count := 0;
      for I in Export_Index range 1 .. Max_Exports loop
         IR.Exports.Exports (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Exports.Exports (I).Export_Type := Type_Name_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize types
      IR.Types.Count := 0;
      for I in Type_Def_Index range 1 .. Max_Type_Defs loop
         IR.Types.Type_Defs (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Types.Type_Defs (I).Kind := Type_Alias;
         IR.Types.Type_Defs (I).Base_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Types.Type_Defs (I).Fields.Count := 0;
      end loop;
      
      --  Initialize constants
      IR.Constants.Count := 0;
      for I in Constant_Index range 1 .. Max_Constants loop
         IR.Constants.Constants (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Constants.Constants (I).Const_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Constants.Constants (I).Value_Str := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize dependencies
      IR.Dependencies.Count := 0;
      for I in Dependency_Index range 1 .. Max_Dependencies loop
         IR.Dependencies.Dependencies (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Dependencies.Dependencies (I).Version := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      IR.Functions.Count := 0;
      
      --  Initialize functions array with defaults
      for I in Function_Index range 1 .. Max_Functions loop
         IR.Functions.Functions (I).Name := Identifier_Strings.Null_Bounded_String;
         IR.Functions.Functions (I).Return_Type := Type_Name_Strings.Null_Bounded_String;
         IR.Functions.Functions (I).Parameters.Count := 0;
         IR.Functions.Functions (I).Steps.Count := 0;
         
         --  Initialize steps with default values
         for J in Step_Index range 1 .. Max_Steps loop
            IR.Functions.Functions (I).Steps.Steps (J) := Make_Default_Step;
         end loop;
      end loop;
   end Init_IR_Data;

   procedure Parse_IR_String
     (JSON_Content : in     JSON_String;
      IR           :    out IR_Data;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Initialize IR with defaults
      Init_IR_Data (IR);

      --  Use simple JSON parsing to extract module name
      Initialize_Parser (Parser, JSON_Content, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Parse;
         return;
      end if;

      --  Expect object start
      Next_Token (Parser, Temp_Status);
      if Temp_Status /= Success or else
         Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      --  Parse root object members
      Next_Token (Parser, Temp_Status);
      while Temp_Status = Success and then
            Current_Token (Parser) /= Token_Object_End loop
         declare
            Member_Name  : Identifier_String;
            Member_Value : JSON_String;
         begin
            Parse_String_Member (Parser, Member_Name, Member_Value, Temp_Status);
            if Temp_Status /= Success then
               Status := Error_Parse;
               return;
            end if;

            declare
               Name_Str : constant String := Identifier_Strings.To_String (Member_Name);
            begin
               if Name_Str = "module_name" then
                  IR.Module_Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "ir_version" then
                  IR.IR_Version := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "schema_version" or Name_Str = "schema" then
                  IR.Schema_Version := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "imports" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse imports array
                  IR.Imports.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Imports.Count < Max_Imports then
                           IR.Imports.Count := IR.Imports.Count + 1;
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Imp_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    if Imp_Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Imports.Imports (IR.Imports.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Imp_Key = "from" and then Current_Token (Parser) = Token_String then
                                       IR.Imports.Imports (IR.Imports.Count).From_Module :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    else
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "exports" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse exports array
                  IR.Exports.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Exports.Count < Max_Exports then
                           IR.Exports.Count := IR.Exports.Count + 1;
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Exp_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    if Exp_Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Exports.Exports (IR.Exports.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Exp_Key = "type" and then Current_Token (Parser) = Token_String then
                                       IR.Exports.Exports (IR.Exports.Count).Export_Type :=
                                          Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    else
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "types" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse types array
                  IR.Types.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Types.Count < Max_Type_Defs then
                           IR.Types.Count := IR.Types.Count + 1;
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Type_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    if Type_Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Types.Type_Defs (IR.Types.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Type_Key = "kind" and then Current_Token (Parser) = Token_String then
                                       declare
                                          Kind_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                       begin
                                          if Kind_Str = "struct" then
                                             IR.Types.Type_Defs (IR.Types.Count).Kind := Type_Struct;
                                          elsif Kind_Str = "enum" then
                                             IR.Types.Type_Defs (IR.Types.Count).Kind := Type_Enum;
                                          elsif Kind_Str = "alias" then
                                             IR.Types.Type_Defs (IR.Types.Count).Kind := Type_Alias;
                                          elsif Kind_Str = "generic" then
                                             IR.Types.Type_Defs (IR.Types.Count).Kind := Type_Generic;
                                          end if;
                                       end;
                                       Next_Token (Parser, Temp_Status);
                                    elsif Type_Key = "base_type" and then Current_Token (Parser) = Token_String then
                                       IR.Types.Type_Defs (IR.Types.Count).Base_Type :=
                                          Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Type_Key = "fields" and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse fields array
                                       IR.Types.Type_Defs (IR.Types.Count).Fields.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             if IR.Types.Type_Defs (IR.Types.Count).Fields.Count < Max_Type_Fields then
                                                IR.Types.Type_Defs (IR.Types.Count).Fields.Count :=
                                                   IR.Types.Type_Defs (IR.Types.Count).Fields.Count + 1;
                                                Next_Token (Parser, Temp_Status);
                                                while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                   if Current_Token (Parser) = Token_String then
                                                      declare
                                                         Field_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         Next_Token (Parser, Temp_Status);
                                                         if Current_Token (Parser) /= Token_Colon then
                                                            Status := Error_Parse;
                                                            return;
                                                         end if;
                                                         Next_Token (Parser, Temp_Status);
                                                         if Field_Key = "name" and then Current_Token (Parser) = Token_String then
                                                            IR.Types.Type_Defs (IR.Types.Count).Fields.Fields (
                                                               IR.Types.Type_Defs (IR.Types.Count).Fields.Count).Name :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Field_Key = "type" and then Current_Token (Parser) = Token_String then
                                                            IR.Types.Type_Defs (IR.Types.Count).Fields.Fields (
                                                               IR.Types.Type_Defs (IR.Types.Count).Fields.Count).Field_Type :=
                                                               Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         else
                                                            Skip_Value (Parser, Temp_Status);
                                                         end if;
                                                      end;
                                                   else
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                   if Current_Token (Parser) = Token_Comma then
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                end loop;
                                                if Current_Token (Parser) = Token_Object_End then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end if;
                                          else
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                          if Current_Token (Parser) = Token_Comma then
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                       end loop;
                                       if Current_Token (Parser) = Token_Array_End then
                                          Next_Token (Parser, Temp_Status);
                                       end if;
                                    else
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "constants" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse constants array
                  IR.Constants.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Constants.Count < Max_Constants then
                           IR.Constants.Count := IR.Constants.Count + 1;
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Const_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    if Const_Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Constants.Constants (IR.Constants.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Const_Key = "type" and then Current_Token (Parser) = Token_String then
                                       IR.Constants.Constants (IR.Constants.Count).Const_Type :=
                                          Type_Name_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Const_Key = "value" and then Current_Token (Parser) = Token_String then
                                       IR.Constants.Constants (IR.Constants.Count).Value_Str :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    else
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "dependencies" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse dependencies array
                  IR.Dependencies.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Dependencies.Count < Max_Dependencies then
                           IR.Dependencies.Count := IR.Dependencies.Count + 1;
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Dep_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    if Dep_Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Dependencies.Dependencies (IR.Dependencies.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Dep_Key = "version" and then Current_Token (Parser) = Token_String then
                                       IR.Dependencies.Dependencies (IR.Dependencies.Count).Version :=
                                          Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    else
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                              if Current_Token (Parser) = Token_Comma then
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "functions" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse functions array
                  IR.Functions.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if IR.Functions.Count < Max_Functions then
                           IR.Functions.Count := IR.Functions.Count + 1;
                           
                           --  Parse function object members
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              --  Expect string key
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    --  Expect colon
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    
                                    Next_Token (Parser, Temp_Status);
                                    
                                    if Key = "name" and then Current_Token (Parser) = Token_String then
                                       IR.Functions.Functions (IR.Functions.Count).Name := 
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Key = "return_type" and then Current_Token (Parser) = Token_String then
                                       IR.Functions.Functions (IR.Functions.Count).Return_Type := 
                                          Type_Name_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Key = "args" and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse args array
                                       IR.Functions.Functions (IR.Functions.Count).Parameters.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             if IR.Functions.Functions (IR.Functions.Count).Parameters.Count < Max_Parameters then
                                                IR.Functions.Functions (IR.Functions.Count).Parameters.Count :=
                                                   IR.Functions.Functions (IR.Functions.Count).Parameters.Count + 1;
                                                
                                                --  Parse arg object
                                                Next_Token (Parser, Temp_Status);
                                                while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                   if Current_Token (Parser) = Token_String then
                                                      declare
                                                         Arg_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         Next_Token (Parser, Temp_Status);
                                                         if Current_Token (Parser) /= Token_Colon then
                                                            Status := Error_Parse;
                                                            return;
                                                         end if;
                                                         Next_Token (Parser, Temp_Status);
                                                         
                                                         if Arg_Key = "name" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Parameters.Params (
                                                               IR.Functions.Functions (IR.Functions.Count).Parameters.Count).Name :=
                                                               Identifier_Strings.To_Bounded_String (
                                                                  JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Arg_Key = "type" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Parameters.Params (
                                                               IR.Functions.Functions (IR.Functions.Count).Parameters.Count).Param_Type :=
                                                               Type_Name_Strings.To_Bounded_String (
                                                                  JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         else
                                                            Skip_Value (Parser, Temp_Status);
                                                         end if;
                                                      end;
                                                   else
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                   if Current_Token (Parser) = Token_Comma then
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                end loop;
                                                if Current_Token (Parser) = Token_Object_End then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end if;
                                          else
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                          if Current_Token (Parser) = Token_Comma then
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                       end loop;
                                       if Current_Token (Parser) = Token_Array_End then
                                          Next_Token (Parser, Temp_Status);
                                       end if;
                                    elsif Key = "steps" and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse steps array
                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             if IR.Functions.Functions (IR.Functions.Count).Steps.Count < Max_Steps then
                                                IR.Functions.Functions (IR.Functions.Count).Steps.Count :=
                                                   IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                
                                                --  Parse step object
                                                Next_Token (Parser, Temp_Status);
                                                while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                   if Current_Token (Parser) = Token_String then
                                                      declare
                                                         Step_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         Next_Token (Parser, Temp_Status);
                                                         if Current_Token (Parser) /= Token_Colon then
                                                            Status := Error_Parse;
                                                            return;
                                                         end if;
                                                         Next_Token (Parser, Temp_Status);
                                                         
                                                         if Step_Key = "op" and then Current_Token (Parser) = Token_String then
                                                            declare
                                                               Op_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                            begin
                                                               if Op_Str = "return" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_Return;
                                                               elsif Op_Str = "assign" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_Assign;
                                                               elsif Op_Str = "call" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_Call;
                                                               elsif Op_Str = "if" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_If;
                                                               elsif Op_Str = "while" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_While;
                                                               elsif Op_Str = "for" then
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_For;
                                                               else
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Count).Step_Type := Step_Nop;
                                                               end if;
                                                            end;
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "value" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Value :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "target" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Target :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "condition" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Condition :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "init" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Init :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "increment" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Increment :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "args" and then Current_Token (Parser) = Token_String then
                                                            IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Count).Args :=
                                                               Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Step_Key = "then_steps" and then Current_Token (Parser) = Token_Array_Start then
                                                            --  Parse then_steps array for if blocks
                                                            declare
                                                               Then_Start : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                               Then_Count : Step_Index := 0;
                                                            begin
                                                               Next_Token (Parser, Temp_Status);
                                                               while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                  if Current_Token (Parser) = Token_Object_Start then
                                                                     if IR.Functions.Functions (IR.Functions.Count).Steps.Count < Max_Steps then
                                                                        IR.Functions.Functions (IR.Functions.Count).Steps.Count :=
                                                                           IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                                        Then_Count := Then_Count + 1;
                                                                        --  Parse nested step object (simplified - just op and value)
                                                                        Next_Token (Parser, Temp_Status);
                                                                        while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                           if Current_Token (Parser) = Token_String then
                                                                              declare
                                                                                 Nested_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Current_Token (Parser) /= Token_Colon then
                                                                                    Status := Error_Parse;
                                                                                    return;
                                                                                 end if;
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Nested_Key = "op" and then Current_Token (Parser) = Token_String then
                                                                                    declare
                                                                                       Op_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                       Step_Idx : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count;
                                                                                    begin
                                                                                       if Op_Str = "return" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Return;
                                                                                       elsif Op_Str = "assign" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Assign;
                                                                                       elsif Op_Str = "call" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Call;
                                                                                       else
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Nop;
                                                                                       end if;
                                                                                    end;
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "value" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Value :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Target :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 else
                                                                                    Skip_Value (Parser, Temp_Status);
                                                                                 end if;
                                                                              end;
                                                                           else
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                           if Current_Token (Parser) = Token_Comma then
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                        end loop;
                                                                        if Current_Token (Parser) = Token_Object_End then
                                                                           Next_Token (Parser, Temp_Status);
                                                                        end if;
                                                                     end if;
                                                                  else
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                                  if Current_Token (Parser) = Token_Comma then
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                               end loop;
                                                               --  Store then block indices in the parent if step
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Count - Then_Count).Then_Start := Then_Start;
                                                               IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                  IR.Functions.Functions (IR.Functions.Count).Steps.Count - Then_Count).Then_Count := Then_Count;
                                                               if Current_Token (Parser) = Token_Array_End then
                                                                  Next_Token (Parser, Temp_Status);
                                                               end if;
                                                            end;
                                                         elsif Step_Key = "else_steps" and then Current_Token (Parser) = Token_Array_Start then
                                                            --  Parse else_steps array for if blocks
                                                            declare
                                                               Else_Start : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                               Else_Count : Step_Index := 0;
                                                            begin
                                                               Next_Token (Parser, Temp_Status);
                                                               while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                  if Current_Token (Parser) = Token_Object_Start then
                                                                     if IR.Functions.Functions (IR.Functions.Count).Steps.Count < Max_Steps then
                                                                        IR.Functions.Functions (IR.Functions.Count).Steps.Count :=
                                                                           IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                                        Else_Count := Else_Count + 1;
                                                                        --  Parse nested step object (simplified)
                                                                        Next_Token (Parser, Temp_Status);
                                                                        while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                           if Current_Token (Parser) = Token_String then
                                                                              declare
                                                                                 Nested_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Current_Token (Parser) /= Token_Colon then
                                                                                    Status := Error_Parse;
                                                                                    return;
                                                                                 end if;
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Nested_Key = "op" and then Current_Token (Parser) = Token_String then
                                                                                    declare
                                                                                       Op_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                       Step_Idx : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count;
                                                                                    begin
                                                                                       if Op_Str = "return" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Return;
                                                                                       elsif Op_Str = "assign" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Assign;
                                                                                       elsif Op_Str = "call" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Call;
                                                                                       else
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Nop;
                                                                                       end if;
                                                                                    end;
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "value" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Value :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Target :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 else
                                                                                    Skip_Value (Parser, Temp_Status);
                                                                                 end if;
                                                                              end;
                                                                           else
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                           if Current_Token (Parser) = Token_Comma then
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                        end loop;
                                                                        if Current_Token (Parser) = Token_Object_End then
                                                                           Next_Token (Parser, Temp_Status);
                                                                        end if;
                                                                     end if;
                                                                  else
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                                  if Current_Token (Parser) = Token_Comma then
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                               end loop;
                                                               --  Store else block indices in the parent if step (find the if step)
                                                               for Find_If in reverse Step_Index range 1 .. IR.Functions.Functions (IR.Functions.Count).Steps.Count loop
                                                                  if IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_If).Step_Type = Step_If then
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_If).Else_Start := Else_Start;
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_If).Else_Count := Else_Count;
                                                                     exit;
                                                                  end if;
                                                               end loop;
                                                               if Current_Token (Parser) = Token_Array_End then
                                                                  Next_Token (Parser, Temp_Status);
                                                               end if;
                                                            end;
                                                         elsif Step_Key = "body_steps" and then Current_Token (Parser) = Token_Array_Start then
                                                            --  Parse body_steps array for while/for blocks
                                                            declare
                                                               Body_Start : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                               Body_Count : Step_Index := 0;
                                                            begin
                                                               Next_Token (Parser, Temp_Status);
                                                               while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                  if Current_Token (Parser) = Token_Object_Start then
                                                                     if IR.Functions.Functions (IR.Functions.Count).Steps.Count < Max_Steps then
                                                                        IR.Functions.Functions (IR.Functions.Count).Steps.Count :=
                                                                           IR.Functions.Functions (IR.Functions.Count).Steps.Count + 1;
                                                                        Body_Count := Body_Count + 1;
                                                                        --  Parse nested step object (simplified)
                                                                        Next_Token (Parser, Temp_Status);
                                                                        while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                           if Current_Token (Parser) = Token_String then
                                                                              declare
                                                                                 Nested_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Current_Token (Parser) /= Token_Colon then
                                                                                    Status := Error_Parse;
                                                                                    return;
                                                                                 end if;
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 if Nested_Key = "op" and then Current_Token (Parser) = Token_String then
                                                                                    declare
                                                                                       Op_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                       Step_Idx : constant Step_Index := IR.Functions.Functions (IR.Functions.Count).Steps.Count;
                                                                                    begin
                                                                                       if Op_Str = "return" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Return;
                                                                                       elsif Op_Str = "assign" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Assign;
                                                                                       elsif Op_Str = "call" then
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Call;
                                                                                       else
                                                                                          IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Step_Idx).Step_Type := Step_Nop;
                                                                                       end if;
                                                                                    end;
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "value" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Value :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 elsif Nested_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                                    IR.Functions.Functions (IR.Functions.Count).Steps.Steps (
                                                                                       IR.Functions.Functions (IR.Functions.Count).Steps.Count).Target :=
                                                                                       Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 else
                                                                                    Skip_Value (Parser, Temp_Status);
                                                                                 end if;
                                                                              end;
                                                                           else
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                           if Current_Token (Parser) = Token_Comma then
                                                                              Next_Token (Parser, Temp_Status);
                                                                           end if;
                                                                        end loop;
                                                                        if Current_Token (Parser) = Token_Object_End then
                                                                           Next_Token (Parser, Temp_Status);
                                                                        end if;
                                                                     end if;
                                                                  else
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                                  if Current_Token (Parser) = Token_Comma then
                                                                     Next_Token (Parser, Temp_Status);
                                                                  end if;
                                                               end loop;
                                                               --  Store body block indices in the parent while/for step (find it)
                                                               for Find_Loop in reverse Step_Index range 1 .. IR.Functions.Functions (IR.Functions.Count).Steps.Count loop
                                                                  if IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_Loop).Step_Type = Step_While or else
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_Loop).Step_Type = Step_For
                                                                  then
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_Loop).Body_Start := Body_Start;
                                                                     IR.Functions.Functions (IR.Functions.Count).Steps.Steps (Find_Loop).Body_Count := Body_Count;
                                                                     exit;
                                                                  end if;
                                                               end loop;
                                                               if Current_Token (Parser) = Token_Array_End then
                                                                  Next_Token (Parser, Temp_Status);
                                                               end if;
                                                            end;
                                                         else
                                                            Skip_Value (Parser, Temp_Status);
                                                         end if;
                                                      end;
                                                   else
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                   if Current_Token (Parser) = Token_Comma then
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                end loop;
                                                if Current_Token (Parser) = Token_Object_End then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end if;
                                          else
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                          if Current_Token (Parser) = Token_Comma then
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                       end loop;
                                       if Current_Token (Parser) = Token_Array_End then
                                          Next_Token (Parser, Temp_Status);
                                       end if;
                                    else
                                       --  Skip value (unknown)
                                       Skip_Value (Parser, Temp_Status);
                                    end if;
                                    
                                    if Current_Token (Parser) = Token_Comma then
                                       Next_Token (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           
                           --  Expect object end
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                        else
                           Status := Error_Too_Large;
                           return;
                        end if;
                     else
                        Next_Token (Parser, Temp_Status);
                     end if;
                     
                     if Current_Token (Parser) = Token_Comma then
                        Next_Token (Parser, Temp_Status);
                     end if;
                  end loop;
                  if Current_Token (Parser) = Token_Array_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               else
                  --  Unknown member - skip its value
                  if Current_Token (Parser) = Token_Array_Start
                     or Current_Token (Parser) = Token_Object_Start
                  then
                     Skip_Value (Parser, Temp_Status);
                  else
                     Next_Token (Parser, Temp_Status);
                  end if;
               end if;
            end;

            --  Check for comma
            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Temp_Status);
            end if;
         end;
      end loop;

      --  Set defaults if not found
      if Identifier_Strings.Length (IR.Module_Name) = 0 then
         IR.Module_Name := Identifier_Strings.To_Bounded_String ("module");
      end if;
      if Identifier_Strings.Length (IR.IR_Version) = 0 then
         IR.IR_Version := Identifier_Strings.To_Bounded_String ("1.0");
      end if;

      Status := Success;
   end Parse_IR_String;

   procedure Parse_IR_File
     (Input_Path : in     Path_String;
      IR         :    out IR_Data;
      Status     :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK
      
      Input_File   : File_Type;
      File_Content : String (1 .. Max_JSON_Length);
      Content_Len  : Natural := 0;
      JSON_Content : JSON_String;
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
                  Status := Error_Too_Large;
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
      
      --  Parse the JSON string
      Parse_IR_String (JSON_Content, IR, Status);
   end Parse_IR_File;

end IR_Parse;
