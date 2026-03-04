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
                              --  Skip types array for now (complex parsing)
                              Skip_Value (Parser, Temp_Status);
                           elsif Mod_Key = "constants" and then Current_Token (Parser) = Token_Array_Start then
                              --  Skip constants array for now (complex parsing)
                              Skip_Value (Parser, Temp_Status);
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
