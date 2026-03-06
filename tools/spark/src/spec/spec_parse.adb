--  Spec Parse Micro-Tool Body
--  Parses spec JSON into internal representation
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;
with Ada.Text_IO;
use Ada.Text_IO;

package body Spec_Parse is

   --  Internal helper to initialize Spec_Data with defaults
   procedure Init_Spec_Data (Spec : out Spec_Data) is
   begin
      Spec.Schema_Version := Identifier_Strings.Null_Bounded_String;
      Spec.Module_Name    := Identifier_Strings.Null_Bounded_String;
      
      --  Initialize imports
      Spec.Imports.Count := 0;
      for I in Import_Index range 1 .. Max_Imports loop
         Spec.Imports.Imports (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Imports.Imports (I).From_Module := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize exports
      Spec.Exports.Count := 0;
      for I in Export_Index range 1 .. Max_Exports loop
         Spec.Exports.Exports (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Exports.Exports (I).Export_Type := Type_Name_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize types
      Spec.Types.Count := 0;
      for I in Type_Def_Index range 1 .. Max_Type_Defs loop
         Spec.Types.Type_Defs (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Types.Type_Defs (I).Base_Type := Type_Name_Strings.Null_Bounded_String;
         Spec.Types.Type_Defs (I).Fields.Count := 0;
      end loop;
      
      --  Initialize constants
      Spec.Constants.Count := 0;
      for I in Constant_Index range 1 .. Max_Constants loop
         Spec.Constants.Constants (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Constants.Constants (I).Const_Type := Type_Name_Strings.Null_Bounded_String;
         Spec.Constants.Constants (I).Value_Str := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize dependencies
      Spec.Dependencies.Count := 0;
      for I in Dependency_Index range 1 .. Max_Dependencies loop
         Spec.Dependencies.Dependencies (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Dependencies.Dependencies (I).Version := Identifier_Strings.Null_Bounded_String;
      end loop;
      
      --  Initialize functions
      Spec.Functions.Count := 0;
      for I in Function_Index range 1 .. Max_Functions loop
         Spec.Functions.Functions (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Functions.Functions (I).Return_Type := Type_Name_Strings.Null_Bounded_String;
         Spec.Functions.Functions (I).Parameters.Count := 0;
         Spec.Functions.Functions (I).Stmts.Count := 0;
         Spec.Functions.Functions (I).Body_Hint := Hint_None;
         Spec.Functions.Functions (I).Hint_Detail := Hint_Strings.Null_Bounded_String;
         for J in 1 .. Max_Statements loop
            Spec.Functions.Functions (I).Stmts.Statements (J).Stmt_Type := Stmt_Nop;
            Spec.Functions.Functions (I).Stmts.Statements (J).Target := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Value := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Condition := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Init := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Increment := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Expr := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Args := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Error_Msg := Identifier_Strings.Null_Bounded_String;
            Spec.Functions.Functions (I).Stmts.Statements (J).Case_Count := 0;
            Spec.Functions.Functions (I).Stmts.Statements (J).Has_Default := False;
         end loop;
      end loop;
      
      --  Initialize artifacts
      Spec.Precompiled.GPU_Binaries.Count := 0;
      for I in GPU_Binary_Index range 1 .. Max_GPU_Binaries loop
         Spec.Precompiled.GPU_Binaries.Binaries (I).Format := Format_PTX;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Digest := Identifier_Strings.Null_Bounded_String;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Target_Arch := Identifier_Strings.Null_Bounded_String;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Entry_Count := 0;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Blob_Path := Path_Strings.Null_Bounded_String;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Kernel_Name := Identifier_Strings.Null_Bounded_String;
         Spec.Precompiled.GPU_Binaries.Binaries (I).Policy := Prefer_Source;
      end loop;
      
      Spec.Precompiled.Microcode_Blobs.Count := 0;
      for I in Microcode_Blob_Index range 1 .. Max_Microcode_Blobs loop
         Spec.Precompiled.Microcode_Blobs.Blobs (I).Format := Format_Microcode;
         Spec.Precompiled.Microcode_Blobs.Blobs (I).Digest := Identifier_Strings.Null_Bounded_String;
         Spec.Precompiled.Microcode_Blobs.Blobs (I).Target_Device := Identifier_Strings.Null_Bounded_String;
         Spec.Precompiled.Microcode_Blobs.Blobs (I).Blob_Path := Path_Strings.Null_Bounded_String;
         Spec.Precompiled.Microcode_Blobs.Blobs (I).Load_Address := Identifier_Strings.Null_Bounded_String;
      end loop;
   end Init_Spec_Data;

   procedure Parse_Spec_String
     (JSON_Content : in     JSON_String;
      Spec         :    out Spec_Data;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Initialize Spec with defaults
      Init_Spec_Data (Spec);

      --  Use simple JSON parsing
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
               if Name_Str = "module_name" or Name_Str = "name" then
                  Spec.Module_Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "schema_version" or Name_Str = "schema" then
                  Spec.Schema_Version := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "dependencies" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse dependencies array
                  Spec.Dependencies.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if Spec.Dependencies.Count < Max_Dependencies then
                           Spec.Dependencies.Count := Spec.Dependencies.Count + 1;
                           
                           --  Parse dependency object members
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Dep_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                    Dep_Idx : constant Dependency_Index := Spec.Dependencies.Count;
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    
                                    if Dep_Key = "name" and then Current_Token (Parser) = Token_String then
                                       Spec.Dependencies.Dependencies (Dep_Idx).Name :=
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Dep_Key = "version" and then Current_Token (Parser) = Token_String then
                                       Spec.Dependencies.Dependencies (Dep_Idx).Version :=
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    else
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
               elsif Name_Str = "module" and then Current_Token (Parser) = Token_Object_Start then
                  --  Parse module object
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                     if Current_Token (Parser) = Token_String then
                        declare
                           Mod_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                        begin
                           Next_Token (Parser, Temp_Status);
                           if Current_Token (Parser) /= Token_Colon then
                              Status := Error_Parse;
                              return;
                           end if;
                           Next_Token (Parser, Temp_Status);
                           
                           if Mod_Key = "name" and then Current_Token (Parser) = Token_String then
                              Spec.Module_Name := Identifier_Strings.To_Bounded_String (
                                 JSON_Strings.To_String (Token_String_Value (Parser)));
                              Next_Token (Parser, Temp_Status);
                           elsif Mod_Key = "imports" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse imports array (simplified: string array)
                              Spec.Imports.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_String then
                                    if Spec.Imports.Count < Max_Imports then
                                       Spec.Imports.Count := Spec.Imports.Count + 1;
                                       Spec.Imports.Imports (Spec.Imports.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                    end if;
                                    Next_Token (Parser, Temp_Status);
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
                           elsif Mod_Key = "exports" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse exports array (simplified: string array)
                              Spec.Exports.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_String then
                                    if Spec.Exports.Count < Max_Exports then
                                       Spec.Exports.Count := Spec.Exports.Count + 1;
                                       Spec.Exports.Exports (Spec.Exports.Count).Name :=
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                    end if;
                                    Next_Token (Parser, Temp_Status);
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
                           elsif Mod_Key = "types" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse types array
                              Spec.Types.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_Object_Start then
                                    if Spec.Types.Count < Max_Type_Defs then
                                       Spec.Types.Count := Spec.Types.Count + 1;
                                       
                                       --  Initialize type definition
                                       Spec.Types.Type_Defs (Spec.Types.Count).Kind := Type_Struct;
                                       Spec.Types.Type_Defs (Spec.Types.Count).Fields.Count := 0;
                                       Spec.Types.Type_Defs (Spec.Types.Count).Base_Type := 
                                          Type_Name_Strings.Null_Bounded_String;
                                       
                                       --  Parse type object members
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                          if Current_Token (Parser) = Token_String then
                                             declare
                                                Type_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                Type_Idx : constant Type_Def_Index := Spec.Types.Count;
                                             begin
                                                Next_Token (Parser, Temp_Status);
                                                if Current_Token (Parser) /= Token_Colon then
                                                   Status := Error_Parse;
                                                   return;
                                                end if;
                                                Next_Token (Parser, Temp_Status);
                                                
                                                if Type_Key = "name" and then Current_Token (Parser) = Token_String then
                                                   Spec.Types.Type_Defs (Type_Idx).Name := 
                                                      Identifier_Strings.To_Bounded_String (
                                                         JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Type_Key = "kind" and then Current_Token (Parser) = Token_String then
                                                   declare
                                                      Kind_Str : constant String := 
                                                         JSON_Strings.To_String (Token_String_Value (Parser));
                                                   begin
                                                      if Kind_Str = "struct" then
                                                         Spec.Types.Type_Defs (Type_Idx).Kind := Type_Struct;
                                                      elsif Kind_Str = "enum" then
                                                         Spec.Types.Type_Defs (Type_Idx).Kind := Type_Enum;
                                                      elsif Kind_Str = "alias" then
                                                         Spec.Types.Type_Defs (Type_Idx).Kind := Type_Alias;
                                                      elsif Kind_Str = "generic" then
                                                         Spec.Types.Type_Defs (Type_Idx).Kind := Type_Generic;
                                                      end if;
                                                   end;
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Type_Key = "base_type" and then Current_Token (Parser) = Token_String then
                                                   Spec.Types.Type_Defs (Type_Idx).Base_Type := 
                                                      Type_Name_Strings.To_Bounded_String (
                                                         JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Type_Key = "fields" and then Current_Token (Parser) = Token_Array_Start then
                                                   --  Parse fields array for struct types
                                                   Spec.Types.Type_Defs (Type_Idx).Fields.Count := 0;
                                                   Next_Token (Parser, Temp_Status);
                                                   while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                      if Current_Token (Parser) = Token_Object_Start then
                                                         if Spec.Types.Type_Defs (Type_Idx).Fields.Count < Max_Type_Fields then
                                                            Spec.Types.Type_Defs (Type_Idx).Fields.Count := 
                                                               Spec.Types.Type_Defs (Type_Idx).Fields.Count + 1;
                                                            
                                                            --  Parse field object
                                                            Next_Token (Parser, Temp_Status);
                                                            while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                               if Current_Token (Parser) = Token_String then
                                                                  declare
                                                                     Field_Key : constant String := 
                                                                        JSON_Strings.To_String (Token_String_Value (Parser));
                                                                     Field_Idx : constant Type_Field_Index := 
                                                                        Spec.Types.Type_Defs (Type_Idx).Fields.Count;
                                                                  begin
                                                                     Next_Token (Parser, Temp_Status);
                                                                     if Current_Token (Parser) /= Token_Colon then
                                                                        Status := Error_Parse;
                                                                        return;
                                                                     end if;
                                                                     Next_Token (Parser, Temp_Status);
                                                                     
                                                                     if Field_Key = "name" and then Current_Token (Parser) = Token_String then
                                                                        Spec.Types.Type_Defs (Type_Idx).Fields.Fields (Field_Idx).Name :=
                                                                           Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                        Next_Token (Parser, Temp_Status);
                                                                     elsif Field_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                        Spec.Types.Type_Defs (Type_Idx).Fields.Fields (Field_Idx).Field_Type :=
                                                                           Type_Name_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                        Next_Token (Parser, Temp_Status);
                                                                     else
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
                                                
                                                if Current_Token (Parser) = Token_Comma then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end;
                                          else
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
                           elsif Mod_Key = "constants" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse constants array
                              Spec.Constants.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_Object_Start then
                                    if Spec.Constants.Count < Max_Constants then
                                       Spec.Constants.Count := Spec.Constants.Count + 1;
                                       
                                       --  Parse constant object members
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                          if Current_Token (Parser) = Token_String then
                                             declare
                                                Const_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                Const_Idx : constant Constant_Index := Spec.Constants.Count;
                                             begin
                                                Next_Token (Parser, Temp_Status);
                                                if Current_Token (Parser) /= Token_Colon then
                                                   Status := Error_Parse;
                                                   return;
                                                end if;
                                                Next_Token (Parser, Temp_Status);
                                                
                                                if Const_Key = "name" and then Current_Token (Parser) = Token_String then
                                                   Spec.Constants.Constants (Const_Idx).Name := 
                                                      Identifier_Strings.To_Bounded_String (
                                                         JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Const_Key = "type" and then Current_Token (Parser) = Token_String then
                                                   Spec.Constants.Constants (Const_Idx).Const_Type := 
                                                      Type_Name_Strings.To_Bounded_String (
                                                         JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Const_Key = "value" then
                                                   --  Parse value (can be string, number, boolean, etc.)
                                                   if Current_Token (Parser) = Token_String then
                                                      Spec.Constants.Constants (Const_Idx).Value_Str := 
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif Current_Token (Parser) = Token_Number then
                                                      Spec.Constants.Constants (Const_Idx).Value_Str := 
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif Current_Token (Parser) = Token_True then
                                                      Spec.Constants.Constants (Const_Idx).Value_Str := 
                                                         Identifier_Strings.To_Bounded_String ("true");
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif Current_Token (Parser) = Token_False then
                                                      Spec.Constants.Constants (Const_Idx).Value_Str := 
                                                         Identifier_Strings.To_Bounded_String ("false");
                                                      Next_Token (Parser, Temp_Status);
                                                   elsif Current_Token (Parser) = Token_Null then
                                                      Spec.Constants.Constants (Const_Idx).Value_Str := 
                                                         Identifier_Strings.To_Bounded_String ("null");
                                                      Next_Token (Parser, Temp_Status);
                                                   else
                                                      Skip_Value (Parser, Temp_Status);
                                                   end if;
                                                else
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
                           elsif Mod_Key = "functions" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse functions array (same logic as top-level functions)
                              Spec.Functions.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_Object_Start then
                                    if Spec.Functions.Count < Max_Functions then
                                       Spec.Functions.Count := Spec.Functions.Count + 1;
                                       
                                       --  Parse function object members
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                          if Current_Token (Parser) = Token_String then
                                             declare
                                                Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                             begin
                                                Next_Token (Parser, Temp_Status);
                                                if Current_Token (Parser) /= Token_Colon then
                                                   Status := Error_Parse;
                                                   return;
                                                end if;
                                                Next_Token (Parser, Temp_Status);
                                                
                                                if Key = "name" and then Current_Token (Parser) = Token_String then
                                                   Spec.Functions.Functions (Spec.Functions.Count).Name := 
                                                      Identifier_Strings.To_Bounded_String (
                                                         JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   Next_Token (Parser, Temp_Status);
                                                elsif Key = "return_type" or Key = "returns" then
                                                   if Current_Token (Parser) = Token_String then
                                                      Spec.Functions.Functions (Spec.Functions.Count).Return_Type := 
                                                         Type_Name_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                      Next_Token (Parser, Temp_Status);
                                                   else
                                                      Skip_Value (Parser, Temp_Status);
                                                   end if;
                                                elsif (Key = "args" or Key = "params") and then Current_Token (Parser) = Token_Array_Start then
                                                   --  Parse args array
                                                   Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count := 0;
                                                   Next_Token (Parser, Temp_Status);
                                                   while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                      if Current_Token (Parser) = Token_Object_Start then
                                                         if Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count < Max_Parameters then
                                                            Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count := 
                                                               Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count + 1;
                                                            
                                                            --  Parse arg object
                                                            Next_Token (Parser, Temp_Status);
                                                            while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                               if Current_Token (Parser) = Token_String then
                                                                  declare
                                                                     Arg_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                     Arg_Idx : constant Parameter_Index := 
                                                                        Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count;
                                                                  begin
                                                                     Next_Token (Parser, Temp_Status);
                                                                     if Current_Token (Parser) /= Token_Colon then
                                                                        Status := Error_Parse;
                                                                        return;
                                                                     end if;
                                                                     Next_Token (Parser, Temp_Status);
                                                                     
                                                                     if Arg_Key = "name" and then Current_Token (Parser) = Token_String then
                                                                        Spec.Functions.Functions (Spec.Functions.Count).Parameters.Params (Arg_Idx).Name :=
                                                                           Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                        Next_Token (Parser, Temp_Status);
                                                                     elsif Arg_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                        Spec.Functions.Functions (Spec.Functions.Count).Parameters.Params (Arg_Idx).Param_Type :=
                                                                           Type_Name_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                        Next_Token (Parser, Temp_Status);
                                                                     else
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
                                                elsif Key = "body" and then Current_Token (Parser) = Token_Array_Start then
                                                   --  Parse body array (function statements)
                                                   Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := 0;
                                                   Next_Token (Parser, Temp_Status);
                                                   while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                      if Current_Token (Parser) = Token_Object_Start then
                                                         declare
                                                            Stmt_Idx : constant Natural := 
                                                               Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count + 1;
                                                         begin
                                                            if Stmt_Idx <= Max_Statements then
                                                               Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := Stmt_Idx;
                                                               
                                                               --  Parse statement object
                                                               Next_Token (Parser, Temp_Status);
                                                               while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                  if Current_Token (Parser) = Token_String then
                                                                     declare
                                                                        Stmt_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                        Stmt : Spec_Statement renames 
                                                                           Spec.Functions.Functions (Spec.Functions.Count).Stmts.Statements (Stmt_Idx);
                                                                     begin
                                                                        Next_Token (Parser, Temp_Status);
                                                                        if Current_Token (Parser) /= Token_Colon then
                                                                           Status := Error_Parse;
                                                                           return;
                                                                        end if;
                                                                        Next_Token (Parser, Temp_Status);
                                                                        
                                                                        if Stmt_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                           declare
                                                                              Type_Str : constant String := 
                                                                                 JSON_Strings.To_String (Token_String_Value (Parser));
                                                                           begin
                                                                              if Type_Str = "assign" then
                                                                                 Stmt.Stmt_Type := Stmt_Assign;
                                                                              elsif Type_Str = "return" then
                                                                                 Stmt.Stmt_Type := Stmt_Return;
                                                                              elsif Type_Str = "call" then
                                                                                 Stmt.Stmt_Type := Stmt_Call;
                                                                              elsif Type_Str = "if" then
                                                                                 Stmt.Stmt_Type := Stmt_If;
                                                                              elsif Type_Str = "while" then
                                                                                 Stmt.Stmt_Type := Stmt_While;
                                                                              elsif Type_Str = "for" then
                                                                                 Stmt.Stmt_Type := Stmt_For;
                                                                              elsif Type_Str = "break" then
                                                                                 Stmt.Stmt_Type := Stmt_Break;
                                                                              elsif Type_Str = "continue" then
                                                                                 Stmt.Stmt_Type := Stmt_Continue;
                                                                              elsif Type_Str = "switch" then
                                                                                 Stmt.Stmt_Type := Stmt_Switch;
                                                                              elsif Type_Str = "try" then
                                                                                 Stmt.Stmt_Type := Stmt_Try;
                                                                              elsif Type_Str = "throw" then
                                                                                 Stmt.Stmt_Type := Stmt_Throw;
                                                                              elsif Type_Str = "error" then
                                                                                 Stmt.Stmt_Type := Stmt_Error;
                                                                              else
                                                                                 Stmt.Stmt_Type := Stmt_Nop;
                                                                              end if;
                                                                           end;
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Target := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "value" then
                                                                           if Current_Token (Parser) = Token_String then
                                                                              Stmt.Value := Identifier_Strings.To_Bounded_String (
                                                                                 JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           elsif Current_Token (Parser) = Token_Number then
                                                                              --  Numbers are stored as string representation
                                                                              Stmt.Value := Identifier_Strings.To_Bounded_String (
                                                                                 JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           elsif Current_Token (Parser) = Token_True then
                                                                              Stmt.Value := Identifier_Strings.To_Bounded_String ("true");
                                                                           elsif Current_Token (Parser) = Token_False then
                                                                              Stmt.Value := Identifier_Strings.To_Bounded_String ("false");
                                                                           end if;
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "condition" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Condition := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "init" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Init := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "increment" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Increment := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "expr" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Expr := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                        elsif Stmt_Key = "args" and then Current_Token (Parser) = Token_String then
                                                                           Stmt.Args := Identifier_Strings.To_Bounded_String (
                                                                              JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                           Next_Token (Parser, Temp_Status);
                                                                           elsif Stmt_Key = "body" and then Current_Token (Parser) = Token_Array_Start then
                                                                              --  Parse inline body array and append statements to function Stmts
                                                                              declare
                                                                                 Body_Start : Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count;
                                                                                 Added      : Natural := 0;
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                                    if Current_Token (Parser) = Token_Object_Start then
                                                                                       if Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count < Max_Statements then
                                                                                          declare
                                                                                             New_Idx : constant Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count + 1;
                                                                                          begin
                                                                                             Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := New_Idx;
                                                                                             Added := Added + 1;

                                                                                             --  Parse the nested statement object into the newly appended slot
                                                                                             Next_Token (Parser, Temp_Status);
                                                                                             while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                                                if Current_Token (Parser) = Token_String then
                                                                                                   declare
                                                                                                      Nested_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                                      Nested : Spec_Statement renames
                                                                                                         Spec.Functions.Functions (Spec.Functions.Count).Stmts.Statements (New_Idx);
                                                                                                   begin
                                                                                                      Next_Token (Parser, Temp_Status);
                                                                                                      if Current_Token (Parser) /= Token_Colon then
                                                                                                         Status := Error_Parse;
                                                                                                         return;
                                                                                                      end if;
                                                                                                      Next_Token (Parser, Temp_Status);

                                                                                                      --  Reuse same simple value parsing as top-level statements
                                                                                                      if Nested_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                                                         declare
                                                                                                            Type_Str : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                                         begin
                                                                                                            if Type_Str = "assign" then
                                                                                                               Nested.Stmt_Type := Stmt_Assign;
                                                                                                            elsif Type_Str = "return" then
                                                                                                               Nested.Stmt_Type := Stmt_Return;
                                                                                                            elsif Type_Str = "call" then
                                                                                                               Nested.Stmt_Type := Stmt_Call;
                                                                                                            elsif Type_Str = "if" then
                                                                                                               Nested.Stmt_Type := Stmt_If;
                                                                                                            elsif Type_Str = "while" then
                                                                                                               Nested.Stmt_Type := Stmt_While;
                                                                                                            elsif Type_Str = "for" then
                                                                                                               Nested.Stmt_Type := Stmt_For;
                                                                                                            elsif Type_Str = "break" then
                                                                                                               Nested.Stmt_Type := Stmt_Break;
                                                                                                            elsif Type_Str = "continue" then
                                                                                                               Nested.Stmt_Type := Stmt_Continue;
                                                                                                            elsif Type_Str = "switch" then
                                                                                                               Nested.Stmt_Type := Stmt_Switch;
                                                                                                            elsif Type_Str = "try" then
                                                                                                               Nested.Stmt_Type := Stmt_Try;
                                                                                                            elsif Type_Str = "throw" then
                                                                                                               Nested.Stmt_Type := Stmt_Throw;
                                                                                                            elsif Type_Str = "error" then
                                                                                                               Nested.Stmt_Type := Stmt_Error;
                                                                                                            else
                                                                                                               Nested.Stmt_Type := Stmt_Nop;
                                                                                                            end if;
                                                                                                         end;
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      elsif Nested_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                                                         Nested.Target := Identifier_Strings.To_Bounded_String (
                                                                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      elsif Nested_Key = "value" then
                                                                                                         if Current_Token (Parser) = Token_String then
                                                                                                            Nested.Value := Identifier_Strings.To_Bounded_String (
                                                                                                               JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                                         elsif Current_Token (Parser) = Token_Number then
                                                                                                            Nested.Value := Identifier_Strings.To_Bounded_String (
                                                                                                               JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                                         elsif Current_Token (Parser) = Token_True then
                                                                                                            Nested.Value := Identifier_Strings.To_Bounded_String ("true");
                                                                                                         elsif Current_Token (Parser) = Token_False then
                                                                                                            Nested.Value := Identifier_Strings.To_Bounded_String ("false");
                                                                                                         end if;
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      elsif Nested_Key = "condition" and then Current_Token (Parser) = Token_String then
                                                                                                         Nested.Condition := Identifier_Strings.To_Bounded_String (
                                                                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      else
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

                                                                                             if Current_Token (Parser) = Token_Object_End then
                                                                                                Next_Token (Parser, Temp_Status);
                                                                                             end if;
                                                                                          end;
                                                                                       else
                                                                                          --  Overflow - skip
                                                                                          Skip_Value (Parser, Temp_Status);
                                                                                       end if;
                                                                                    else
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;

                                                                                    if Current_Token (Parser) = Token_Comma then
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;
                                                                                 end loop;

                                                                                 --  Record how many statements were appended for this body's slot
                                                                                 Stmt.Body_Count := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count - Body_Start;
                                                                              end;
                                                                           elsif Stmt_Key = "then" and then Current_Token (Parser) = Token_Array_Start then
                                                                              --  Parse 'then' body for if-statement and append statements
                                                                              declare
                                                                                 Then_Start : Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count;
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                                    if Current_Token (Parser) = Token_Object_Start then
                                                                                       -- reuse the nested-object append logic by jumping to top of loop
                                                                                       -- simple inline parsing similar to 'body' above
                                                                                       if Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count < Max_Statements then
                                                                                          declare
                                                                                             New_Idx : constant Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count + 1;
                                                                                          begin
                                                                                             Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := New_Idx;
                                                                                             Next_Token (Parser, Temp_Status);
                                                                                             while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                                                if Current_Token (Parser) = Token_String then
                                                                                                   declare
                                                                                                      N_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                                      N_Stmt : Spec_Statement renames Spec.Functions.Functions (Spec.Functions.Count).Stmts.Statements (New_Idx);
                                                                                                   begin
                                                                                                      Next_Token (Parser, Temp_Status);
                                                                                                      if Current_Token (Parser) /= Token_Colon then
                                                                                                         Status := Error_Parse;
                                                                                                         return;
                                                                                                      end if;
                                                                                                      Next_Token (Parser, Temp_Status);
                                                                                                      if N_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                                                         declare
                                                                                                            T : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                                         begin
                                                                                                            if T = "assign" then
                                                                                                               N_Stmt.Stmt_Type := Stmt_Assign;
                                                                                                            elsif T = "return" then
                                                                                                               N_Stmt.Stmt_Type := Stmt_Return;
                                                                                                            else
                                                                                                               N_Stmt.Stmt_Type := Stmt_Nop;
                                                                                                            end if;
                                                                                                         end;
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      else
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
                                                                                             if Current_Token (Parser) = Token_Object_End then
                                                                                                Next_Token (Parser, Temp_Status);
                                                                                             end if;
                                                                                          end;
                                                                                       else
                                                                                          Skip_Value (Parser, Temp_Status);
                                                                                       end if;
                                                                                    else
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;
                                                                                    if Current_Token (Parser) = Token_Comma then
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;
                                                                                 end loop;
                                                                                 Stmt.Then_Count := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count - Then_Start;
                                                                              end;
                                                                           elsif Stmt_Key = "else" and then Current_Token (Parser) = Token_Array_Start then
                                                                              --  Parse 'else' body for if-statement and append statements
                                                                              declare
                                                                                 Else_Start : Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count;
                                                                              begin
                                                                                 Next_Token (Parser, Temp_Status);
                                                                                 while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                                    if Current_Token (Parser) = Token_Object_Start then
                                                                                       if Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count < Max_Statements then
                                                                                          declare
                                                                                             New_Idx : constant Natural := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count + 1;
                                                                                          begin
                                                                                             Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := New_Idx;
                                                                                             Next_Token (Parser, Temp_Status);
                                                                                             while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                                                                if Current_Token (Parser) = Token_String then
                                                                                                   declare
                                                                                                      NK : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                                                                      NS : Spec_Statement renames Spec.Functions.Functions (Spec.Functions.Count).Stmts.Statements (New_Idx);
                                                                                                   begin
                                                                                                      Next_Token (Parser, Temp_Status);
                                                                                                      if Current_Token (Parser) /= Token_Colon then
                                                                                                         Status := Error_Parse;
                                                                                                         return;
                                                                                                      end if;
                                                                                                      Next_Token (Parser, Temp_Status);
                                                                                                      if NK = "type" and then Current_Token (Parser) = Token_String then
                                                                                                         NS.Stmt_Type := Stmt_Nop;
                                                                                                         Next_Token (Parser, Temp_Status);
                                                                                                      else
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
                                                                                             if Current_Token (Parser) = Token_Object_End then
                                                                                                Next_Token (Parser, Temp_Status);
                                                                                             end if;
                                                                                          end;
                                                                                       else
                                                                                          Skip_Value (Parser, Temp_Status);
                                                                                       end if;
                                                                                    else
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;
                                                                                    if Current_Token (Parser) = Token_Comma then
                                                                                       Next_Token (Parser, Temp_Status);
                                                                                    end if;
                                                                                 end loop;
                                                                                 Stmt.Else_Count := Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count - Else_Start;
                                                                              end;
                                                                           elsif Stmt_Key = "cases" and then Current_Token (Parser) = Token_Array_Start then
                                                                              --  Parse switch cases; append case bodies into function Stmts and record counts
                                                                              Stmt.Case_Count := 0;
                                                                              Next_Token (Parser, Temp_Status);
                                                                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                                                                 Skip_Value (Parser, Temp_Status);
                                                                                 if Current_Token (Parser) = Token_Comma then
                                                                                    Next_Token (Parser, Temp_Status);
                                                                                 end if;
                                                                              end loop;
                                                                              if Current_Token (Parser) = Token_Array_End then
                                                                                 Next_Token (Parser, Temp_Status);
                                                                              end if;
                                                                           elsif Stmt_Key = "catch" and then Current_Token (Parser) = Token_Array_Start then
                                                                              --  Parse catch blocks; skip for now
                                                                              Skip_Value (Parser, Temp_Status);
                                                                        else
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
                                                               
                                                               if Current_Token (Parser) = Token_Object_End then
                                                                  Next_Token (Parser, Temp_Status);
                                                               end if;
                                                            end if;
                                                         end;
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
                                                
                                                if Current_Token (Parser) = Token_Comma then
                                                   Next_Token (Parser, Temp_Status);
                                                end if;
                                             end;
                                          else
                                             Next_Token (Parser, Temp_Status);
                                          end if;
                                       end loop;
                                       
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
                  if Current_Token (Parser) = Token_Object_End then
                     Next_Token (Parser, Temp_Status);
                  end if;
               elsif Name_Str = "functions" and then Current_Token (Parser) = Token_Array_Start then
                  --  Parse functions array (top-level)
                  Spec.Functions.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if Spec.Functions.Count < Max_Functions then
                           Spec.Functions.Count := Spec.Functions.Count + 1;
                           
                           --  Parse function object members
                           Next_Token (Parser, Temp_Status);
                           while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                              if Current_Token (Parser) = Token_String then
                                 declare
                                    Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                 begin
                                    Next_Token (Parser, Temp_Status);
                                    if Current_Token (Parser) /= Token_Colon then
                                       Status := Error_Parse;
                                       return;
                                    end if;
                                    Next_Token (Parser, Temp_Status);
                                    
                                    if Key = "name" and then Current_Token (Parser) = Token_String then
                                       Spec.Functions.Functions (Spec.Functions.Count).Name := 
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Key = "return_type" or Key = "returns" then
                                       if Current_Token (Parser) = Token_String then
                                          Spec.Functions.Functions (Spec.Functions.Count).Return_Type := 
                                             Type_Name_Strings.To_Bounded_String (
                                                JSON_Strings.To_String (Token_String_Value (Parser)));
                                          Next_Token (Parser, Temp_Status);
                                       else
                                          Skip_Value (Parser, Temp_Status);
                                       end if;
                                    elsif (Key = "args" or Key = "params") and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse args array
                                       Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             if Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count < Max_Parameters then
                                                Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count := 
                                                   Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count + 1;
                                                
                                                --  Parse arg object
                                                Next_Token (Parser, Temp_Status);
                                                while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                   if Current_Token (Parser) = Token_String then
                                                      declare
                                                         Arg_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                         Arg_Idx : constant Parameter_Index := 
                                                            Spec.Functions.Functions (Spec.Functions.Count).Parameters.Count;
                                                      begin
                                                         Next_Token (Parser, Temp_Status);
                                                         if Current_Token (Parser) /= Token_Colon then
                                                            Status := Error_Parse;
                                                            return;
                                                         end if;
                                                         Next_Token (Parser, Temp_Status);
                                                         
                                                         if Arg_Key = "name" and then Current_Token (Parser) = Token_String then
                                                            Spec.Functions.Functions (Spec.Functions.Count).Parameters.Params (Arg_Idx).Name :=
                                                               Identifier_Strings.To_Bounded_String (
                                                                  JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Arg_Key = "type" and then Current_Token (Parser) = Token_String then
                                                            Spec.Functions.Functions (Spec.Functions.Count).Parameters.Params (Arg_Idx).Param_Type :=
                                                               Type_Name_Strings.To_Bounded_String (
                                                                  JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         else
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
                                    elsif Key = "body" and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse body array (function statements)
                                       Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             declare
                                                Stmt_Idx : constant Natural := 
                                                   Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count + 1;
                                             begin
                                                if Stmt_Idx <= Max_Statements then
                                                   Spec.Functions.Functions (Spec.Functions.Count).Stmts.Count := Stmt_Idx;
                                                   
                                                   --  Parse statement object (simplified: only parse type, target, value for now)
                                                   Next_Token (Parser, Temp_Status);
                                                   declare
                                                      Stmt : Spec_Statement renames 
                                                         Spec.Functions.Functions (Spec.Functions.Count).Stmts.Statements (Stmt_Idx);
                                                   begin
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
                                                               
                                                               if Field_Key = "type" and then Current_Token (Parser) = Token_String then
                                                                  declare
                                                                     Type_Str : constant String := 
                                                                        JSON_Strings.To_String (Token_String_Value (Parser));
                                                                  begin
                                                                     if Type_Str = "assign" then
                                                                        Stmt.Stmt_Type := Stmt_Assign;
                                                                     elsif Type_Str = "return" then
                                                                        Stmt.Stmt_Type := Stmt_Return;
                                                                     elsif Type_Str = "call" then
                                                                        Stmt.Stmt_Type := Stmt_Call;
                                                                     elsif Type_Str = "if" then
                                                                        Stmt.Stmt_Type := Stmt_If;
                                                                     elsif Type_Str = "while" then
                                                                        Stmt.Stmt_Type := Stmt_While;
                                                                     elsif Type_Str = "for" then
                                                                        Stmt.Stmt_Type := Stmt_For;
                                                                     elsif Type_Str = "break" then
                                                                        Stmt.Stmt_Type := Stmt_Break;
                                                                     elsif Type_Str = "continue" then
                                                                        Stmt.Stmt_Type := Stmt_Continue;
                                                                     elsif Type_Str = "switch" then
                                                                        Stmt.Stmt_Type := Stmt_Switch;
                                                                     elsif Type_Str = "try" then
                                                                        Stmt.Stmt_Type := Stmt_Try;
                                                                     elsif Type_Str = "throw" then
                                                                        Stmt.Stmt_Type := Stmt_Throw;
                                                                     elsif Type_Str = "error" then
                                                                        Stmt.Stmt_Type := Stmt_Error;
                                                                     else
                                                                        Stmt.Stmt_Type := Stmt_Nop;
                                                                     end if;
                                                                  end;
                                                               elsif Field_Key = "target" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Target := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               elsif Field_Key = "value" then
                                                                  if Current_Token (Parser) = Token_String then
                                                                     Stmt.Value := Identifier_Strings.To_Bounded_String (
                                                                        JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                  elsif Current_Token (Parser) = Token_Number then
                                                                     Stmt.Value := Identifier_Strings.To_Bounded_String (
                                                                        JSON_Strings.To_String (Token_String_Value (Parser)));
                                                                  elsif Current_Token (Parser) = Token_True then
                                                                     Stmt.Value := Identifier_Strings.To_Bounded_String ("true");
                                                                  elsif Current_Token (Parser) = Token_False then
                                                                     Stmt.Value := Identifier_Strings.To_Bounded_String ("false");
                                                                  end if;
                                                               elsif Field_Key = "condition" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Condition := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               elsif Field_Key = "init" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Init := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               elsif Field_Key = "increment" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Increment := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               elsif Field_Key = "expr" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Expr := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               elsif Field_Key = "args" and then Current_Token (Parser) = Token_String then
                                                                  Stmt.Args := Identifier_Strings.To_Bounded_String (
                                                                     JSON_Strings.To_String (Token_String_Value (Parser)));
                                                               else
                                                                  Skip_Value (Parser, Temp_Status);
                                                               end if;
                                                               
                                                               if Current_Token (Parser) /= Token_Object_End then
                                                                  Next_Token (Parser, Temp_Status);
                                                               end if;
                                                            end;
                                                         else
                                                            Next_Token (Parser, Temp_Status);
                                                         end if;
                                                      end loop;
                                                   end;
                                                   
                                                   if Current_Token (Parser) = Token_Object_End then
                                                      Next_Token (Parser, Temp_Status);
                                                   end if;
                                                end if;
                                             end;
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
                                    
                                    if Current_Token (Parser) = Token_Comma then
                                       Next_Token (Parser, Temp_Status);
                                    end if;
                                 end;
                              else
                                 Next_Token (Parser, Temp_Status);
                              end if;
                           end loop;
                           
                           if Current_Token (Parser) = Token_Object_End then
                              Next_Token (Parser, Temp_Status);
                           end if;
                           
                           --  Compute body hint based on parsed statements
                           declare
                              Func_Idx : constant Function_Index := Spec.Functions.Count;
                              Stmt_Count : constant Natural := Spec.Functions.Functions (Func_Idx).Stmts.Count;
                           begin
                              if Stmt_Count = 0 then
                                 --  No body - no hint
                                 Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_None;
                              elsif Stmt_Count = 1 then
                                 --  Single statement - check if it's a simple return
                                 if Spec.Functions.Functions (Func_Idx).Stmts.Statements (1).Stmt_Type = Stmt_Return then
                                    Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Simple_Return;
                                    Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                       Hint_Strings.To_Bounded_String ("single return");
                                 else
                                    Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Simple_Return;
                                 end if;
                              elsif Stmt_Count = 2 then
                                 --  Two statements - likely assign + return
                                 declare
                                    S1 : constant Spec_Statement := 
                                       Spec.Functions.Functions (Func_Idx).Stmts.Statements (1);
                                    S2 : constant Spec_Statement := 
                                       Spec.Functions.Functions (Func_Idx).Stmts.Statements (2);
                                 begin
                                    if S1.Stmt_Type = Stmt_Assign and S2.Stmt_Type = Stmt_Return then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Simple_Return;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("assign and return");
                                    elsif S1.Stmt_Type = Stmt_If then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Conditional;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("simple conditional");
                                    else
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Complex;
                                    end if;
                                 end;
                              else
                                 --  Multiple statements - analyze pattern
                                 declare
                                    Has_Loop : Boolean := False;
                                    Has_If : Boolean := False;
                                    Has_Switch : Boolean := False;
                                    Has_Try : Boolean := False;
                                 begin
                                    for J in 1 .. Stmt_Count loop
                                       case Spec.Functions.Functions (Func_Idx).Stmts.Statements (J).Stmt_Type is
                                          when Stmt_For | Stmt_While =>
                                             Has_Loop := True;
                                          when Stmt_If =>
                                             Has_If := True;
                                          when Stmt_Switch =>
                                             Has_Switch := True;
                                          when Stmt_Try =>
                                             Has_Try := True;
                                          when others =>
                                             null;
                                       end case;
                                    end loop;
                                    
                                    if Has_Loop then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Loop_Accum;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("loop with accumulation");
                                    elsif Has_Switch then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Switch;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("switch/match pattern");
                                    elsif Has_Try then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Try_Catch;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("exception handling");
                                    elsif Has_If then
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Conditional;
                                       Spec.Functions.Functions (Func_Idx).Hint_Detail := 
                                          Hint_Strings.To_Bounded_String ("conditional logic");
                                    else
                                       Spec.Functions.Functions (Func_Idx).Body_Hint := Hint_Complex;
                                    end if;
                                 end;
                              end if;
                           end;
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
               elsif Name_Str = "artifacts" and then Current_Token (Parser) = Token_Object_Start then
                  --  Parse artifacts object
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                     if Current_Token (Parser) = Token_String then
                        declare
                           Art_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                        begin
                           Next_Token (Parser, Temp_Status);
                           if Current_Token (Parser) /= Token_Colon then
                              Status := Error_Parse;
                              return;
                           end if;
                           Next_Token (Parser, Temp_Status);
                           
                           if Art_Key = "gpu_binaries" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse GPU binaries array
                              Spec.Precompiled.GPU_Binaries.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_Object_Start then
                                    if Spec.Precompiled.GPU_Binaries.Count < Max_GPU_Binaries then
                                       Spec.Precompiled.GPU_Binaries.Count := Spec.Precompiled.GPU_Binaries.Count + 1;
                                       declare
                                          Bin_Idx : constant GPU_Binary_Index := Spec.Precompiled.GPU_Binaries.Count;
                                       begin
                                          Next_Token (Parser, Temp_Status);
                                          while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                             if Current_Token (Parser) = Token_String then
                                                declare
                                                   Bin_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                begin
                                                   Next_Token (Parser, Temp_Status);
                                                   if Current_Token (Parser) /= Token_Colon then
                                                      Status := Error_Parse;
                                                      return;
                                                   end if;
                                                   Next_Token (Parser, Temp_Status);
                                                   
                                                   if Bin_Key = "format" and then Current_Token (Parser) = Token_String then
                                                      declare
                                                         Fmt : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         if Fmt = "ptx" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Format := Format_PTX;
                                                         elsif Fmt = "cubin" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Format := Format_CUBIN;
                                                         elsif Fmt = "hsaco" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Format := Format_HSACO;
                                                         elsif Fmt = "spirv" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Format := Format_SPIRV;
                                                         end if;
                                                      end;
                                                   elsif Bin_Key = "digest" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Digest :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Bin_Key = "target_arch" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Target_Arch :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Bin_Key = "blob_path" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Blob_Path :=
                                                         Path_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Bin_Key = "kernel_name" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Kernel_Name :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Bin_Key = "policy" and then Current_Token (Parser) = Token_String then
                                                      declare
                                                         Pol : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         if Pol = "prefer_source" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Policy := Prefer_Source;
                                                         elsif Pol = "prefer_binary" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Policy := Prefer_Binary;
                                                         elsif Pol = "require_binary" then
                                                            Spec.Precompiled.GPU_Binaries.Binaries (Bin_Idx).Policy := Require_Binary;
                                                         end if;
                                                      end;
                                                   end if;
                                                   Next_Token (Parser, Temp_Status);
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
                                       end;
                                    else
                                       Skip_Value (Parser, Temp_Status);
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
                           elsif Art_Key = "microcode_blobs" and then Current_Token (Parser) = Token_Array_Start then
                              --  Parse microcode blobs array
                              Spec.Precompiled.Microcode_Blobs.Count := 0;
                              Next_Token (Parser, Temp_Status);
                              while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                 if Current_Token (Parser) = Token_Object_Start then
                                    if Spec.Precompiled.Microcode_Blobs.Count < Max_Microcode_Blobs then
                                       Spec.Precompiled.Microcode_Blobs.Count := Spec.Precompiled.Microcode_Blobs.Count + 1;
                                       declare
                                          Mc_Idx : constant Microcode_Blob_Index := Spec.Precompiled.Microcode_Blobs.Count;
                                       begin
                                          Next_Token (Parser, Temp_Status);
                                          while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                             if Current_Token (Parser) = Token_String then
                                                declare
                                                   Mc_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                begin
                                                   Next_Token (Parser, Temp_Status);
                                                   if Current_Token (Parser) /= Token_Colon then
                                                      Status := Error_Parse;
                                                      return;
                                                   end if;
                                                   Next_Token (Parser, Temp_Status);
                                                   
                                                   if Mc_Key = "format" and then Current_Token (Parser) = Token_String then
                                                      declare
                                                         Fmt : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                      begin
                                                         if Fmt = "microcode" then
                                                            Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Format := Format_Microcode;
                                                         elsif Fmt = "rom" then
                                                            Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Format := Format_ROM;
                                                         elsif Fmt = "ucode" then
                                                            Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Format := Format_UCode;
                                                         end if;
                                                      end;
                                                   elsif Mc_Key = "digest" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Digest :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Mc_Key = "target_device" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Target_Device :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Mc_Key = "blob_path" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Blob_Path :=
                                                         Path_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   elsif Mc_Key = "load_address" and then Current_Token (Parser) = Token_String then
                                                      Spec.Precompiled.Microcode_Blobs.Blobs (Mc_Idx).Load_Address :=
                                                         Identifier_Strings.To_Bounded_String (
                                                            JSON_Strings.To_String (Token_String_Value (Parser)));
                                                   end if;
                                                   Next_Token (Parser, Temp_Status);
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
                                       end;
                                    else
                                       Skip_Value (Parser, Temp_Status);
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
      if Identifier_Strings.Length (Spec.Module_Name) = 0 then
         Spec.Module_Name := Identifier_Strings.To_Bounded_String ("module");
      end if;

      Status := Success;
   end Parse_Spec_String;

   procedure Parse_Spec_File
     (Input_Path : in     Path_String;
      Spec       :    out Spec_Data;
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
      Parse_Spec_String (JSON_Content, Spec, Status);
   end Parse_Spec_File;

end Spec_Parse;
