--  Extract Parse Micro-Tool Body
--  Parses extraction JSON into internal representation
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;
with Ada.Text_IO;
use Ada.Text_IO;

package body Extract_Parse is

   --  Internal helper to initialize Extract_Data with defaults
   procedure Init_Extract_Data (Extract : out Extract_Data) is
   begin
      Extract.Module_Name := Identifier_Strings.Null_Bounded_String;
      Extract.Functions.Count := 0;
      
      for I in Function_Index range 1 .. Max_Functions loop
         Extract.Functions.Functions (I).Name := Identifier_Strings.Null_Bounded_String;
         Extract.Functions.Functions (I).Return_Type := Type_Name_Strings.Null_Bounded_String;
         Extract.Functions.Functions (I).Parameters.Count := 0;
      end loop;
   end Init_Extract_Data;

   procedure Parse_Extract_String
     (JSON_Content : in     JSON_String;
      Extract      :    out Extract_Data;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      Init_Extract_Data (Extract);

      Initialize_Parser (Parser, JSON_Content, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Parse;
         return;
      end if;

      Next_Token (Parser, Temp_Status);
      if Temp_Status /= Success or else
         Current_Token (Parser) /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

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
                  Extract.Module_Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (Member_Value));
                  Next_Token (Parser, Temp_Status);
               elsif Name_Str = "functions" and then Current_Token (Parser) = Token_Array_Start then
                  Extract.Functions.Count := 0;
                  Next_Token (Parser, Temp_Status);
                  while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                     if Current_Token (Parser) = Token_Object_Start then
                        if Extract.Functions.Count < Max_Functions then
                           Extract.Functions.Count := Extract.Functions.Count + 1;
                           
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
                                       Extract.Functions.Functions (Extract.Functions.Count).Name := 
                                          Identifier_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif Key = "return_type" and then Current_Token (Parser) = Token_String then
                                       Extract.Functions.Functions (Extract.Functions.Count).Return_Type := 
                                          Type_Name_Strings.To_Bounded_String (
                                             JSON_Strings.To_String (Token_String_Value (Parser)));
                                       Next_Token (Parser, Temp_Status);
                                    elsif (Key = "args" or Key = "parameters") and then Current_Token (Parser) = Token_Array_Start then
                                       --  Parse args/parameters array (support both for backward compatibility)
                                       Extract.Functions.Functions (Extract.Functions.Count).Parameters.Count := 0;
                                       Next_Token (Parser, Temp_Status);
                                       while Temp_Status = Success and then Current_Token (Parser) /= Token_Array_End loop
                                          if Current_Token (Parser) = Token_Object_Start then
                                             if Extract.Functions.Functions (Extract.Functions.Count).Parameters.Count < Max_Parameters then
                                                Extract.Functions.Functions (Extract.Functions.Count).Parameters.Count := 
                                                   Extract.Functions.Functions (Extract.Functions.Count).Parameters.Count + 1;
                                                
                                                --  Parse arg object
                                                Next_Token (Parser, Temp_Status);
                                                while Temp_Status = Success and then Current_Token (Parser) /= Token_Object_End loop
                                                   if Current_Token (Parser) = Token_String then
                                                      declare
                                                         Arg_Key : constant String := JSON_Strings.To_String (Token_String_Value (Parser));
                                                         Arg_Idx : constant Parameter_Index := 
                                                            Extract.Functions.Functions (Extract.Functions.Count).Parameters.Count;
                                                      begin
                                                         Next_Token (Parser, Temp_Status);
                                                         if Current_Token (Parser) /= Token_Colon then
                                                            Status := Error_Parse;
                                                            return;
                                                         end if;
                                                         Next_Token (Parser, Temp_Status);
                                                         
                                                         if Arg_Key = "name" and then Current_Token (Parser) = Token_String then
                                                            Extract.Functions.Functions (Extract.Functions.Count).Parameters.Params (Arg_Idx).Name :=
                                                               Identifier_Strings.To_Bounded_String (
                                                                  JSON_Strings.To_String (Token_String_Value (Parser)));
                                                            Next_Token (Parser, Temp_Status);
                                                         elsif Arg_Key = "type" and then Current_Token (Parser) = Token_String then
                                                            Extract.Functions.Functions (Extract.Functions.Count).Parameters.Params (Arg_Idx).Param_Type :=
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
                  if Current_Token (Parser) = Token_Array_Start
                     or Current_Token (Parser) = Token_Object_Start
                  then
                     Skip_Value (Parser, Temp_Status);
                  else
                     Next_Token (Parser, Temp_Status);
                  end if;
               end if;
            end;

            if Current_Token (Parser) = Token_Comma then
               Next_Token (Parser, Temp_Status);
            end if;
         end;
      end loop;

      if Identifier_Strings.Length (Extract.Module_Name) = 0 then
         Extract.Module_Name := Identifier_Strings.To_Bounded_String ("module");
      end if;

      Status := Success;
   end Parse_Extract_String;

   procedure Parse_Extract_File
     (Input_Path : in     Path_String;
      Extract    :    out Extract_Data;
      Status     :    out Status_Code)
   is
      pragma SPARK_Mode (Off);
      
      Input_File   : File_Type;
      File_Content : String (1 .. Max_JSON_Length);
      Content_Len  : Natural := 0;
      JSON_Content : JSON_String;
   begin
      Status := Success;

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
      Parse_Extract_String (JSON_Content, Extract, Status);
   end Parse_Extract_File;

end Extract_Parse;
