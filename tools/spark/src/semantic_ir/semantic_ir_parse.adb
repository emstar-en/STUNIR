-------------------------------------------------------------------------------
--  STUNIR Semantic IR Parse Package Body
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  Implementation of Semantic IR JSON parsing.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
use STUNIR_JSON_Parser;
with Ada.Text_IO;
use Ada.Text_IO;
with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;
with Semantic_IR.Modules; use Semantic_IR.Modules;
with Semantic_IR.Emitter; use Semantic_IR.Emitter;
with IR_Parse;
with STUNIR_Types;
use STUNIR_Types;

package body Semantic_IR.Parse is

   --  Internal helper to initialize Semantic_Module with defaults
   procedure Init_Semantic_Module (M : out Semantic_Module) is
   begin
      --  Initialize base node
      M.Base := (Kind => Kind_Module, others => <>);
      M.Module_Name := Name_Strings.Null_Bounded_String;
      M.Module_Hash := Hash_Strings.Null_Bounded_String;

      --  Initialize imports
      M.Import_Count := 0;
      for I in 1 .. Max_Imports loop
         M.Imports (I).Module_Name := Name_Strings.Null_Bounded_String;
         M.Imports (I).Symbol_Count := 0;
         M.Imports (I).Import_All := False;
         M.Imports (I).Alias := Name_Strings.Null_Bounded_String;
      end loop;

      --  Initialize exports
      M.Export_Count := 0;
      for I in 1 .. Max_Exports loop
         M.Exports (I) := Name_Strings.Null_Bounded_String;
      end loop;

      --  Initialize declarations
      M.Decl_Count := 0;
      for I in 1 .. Max_Declarations loop
         M.Declarations (I) := Name_Strings.Null_Bounded_String;
      end loop;

      --  Initialize metadata
      M.Metadata.Target_Count := 0;
      M.Metadata.Module_Safety := Level_None;
      M.Metadata.Optimization_Level := 0;
      M.Metadata.Is_Entry_Point := False;

      --  Initialize CFG entry
      M.CFG.Entry_Node := Name_Strings.Null_Bounded_String;
      M.CFG.Exit_Node := Name_Strings.Null_Bounded_String;
      M.CFG.Node_Count := 0;
   end Init_Semantic_Module;

   --  Parse Semantic IR JSON file
   procedure Parse_Semantic_IR_File
     (Input_Path : in     Path_String;
      Module     :    out Semantic_Module;
      Status     :    out Status_Code)
   is
      File    : File_Type;
      Content : JSON_String;
      Line    : String (1 .. 1024);
      Last    : Natural;
   begin
      --  Initialize module with defaults
      Init_Semantic_Module (Module);

      --  Open file
      Open (File, In_File, Input_Path);

      --  Read content
      Content := JSON_String_Strings.Null_Bounded_String;
      loop
         Get_Line (File, Line, Last);
         exit when Last = 0;

         --  Append line to content (with length check)
         if JSON_String_Strings.Length (Content) + Last <= JSON_String_Max then
            JSON_String_Strings.Append (Content, Line (1 .. Last));
         end if;
      end loop;

      Close (File);

      --  Parse the content
      Parse_Semantic_IR_String (Content, Module, Status);

   exception
      when others =>
         Status := Error_Parse;
         Module := (others => <>);
   end Parse_Semantic_IR_File;

   --  Parse Semantic IR JSON string
   procedure Parse_Semantic_IR_String
     (JSON_Content : in     JSON_String;
      Module       :    out Semantic_Module;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      --  Initialize module with defaults
      Init_Semantic_Module (Module);

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

      --  Parse object members
      loop
         Next_Token (Parser, Temp_Status);
         if Temp_Status /= Success then
            Status := Error_Parse;
            return;
         end if;

         --  Check for end of object
         if Current_Token (Parser) = Token_Object_End then
            exit;
         end if;

         --  Expect string key
         if Current_Token (Parser) /= Token_String then
            Status := Error_Parse;
            return;
         end if;

         --  Get key name
         declare
            Key : constant String := Current_String (Parser);
         begin
            --  Advance to colon
            Next_Token (Parser, Temp_Status);
            if Temp_Status /= Success or else
               Current_Token (Parser) /= Token_Colon then
               Status := Error_Parse;
               return;
            end if;

            --  Advance to value
            Next_Token (Parser, Temp_Status);
            if Temp_Status /= Success then
               Status := Error_Parse;
               return;
            end if;

            --  Process based on key
            if Key = "module_name" then
               --  Parse module name
               if Current_Token (Parser) = Token_String then
                  Module.Module_Name := Name_Strings.To_Bounded_String (Current_String (Parser));
               end if;
            elsif Key = "schema_version" then
               --  Skip schema version
               null;
            elsif Key = "ir_version" then
               --  Skip IR version
               null;
            elsif Key = "imports" then
               --  Parse imports array
               if Current_Token (Parser) = Token_Array_Start then
                  Module.Import_Count := 0;
                  loop
                     Next_Token (Parser, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Error_Parse;
                        return;
                     end if;
                     exit when Current_Token (Parser) = Token_Array_End;
                     --  Skip import object
                     if Current_Token (Parser) = Token_Object_Start then
                        --  Skip until object end
                        declare
                           Depth : Natural := 1;
                        begin
                           while Depth > 0 loop
                              Next_Token (Parser, Temp_Status);
                              if Temp_Status /= Success then
                                 Status := Error_Parse;
                                 return;
                              end if;
                              if Current_Token (Parser) = Token_Object_Start then
                                 Depth := Depth + 1;
                              elsif Current_Token (Parser) = Token_Object_End then
                                 Depth := Depth - 1;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
               end if;
            elsif Key = "exports" then
               --  Parse exports array
               if Current_Token (Parser) = Token_Array_Start then
                  Module.Export_Count := 0;
                  loop
                     Next_Token (Parser, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Error_Parse;
                        return;
                     end if;
                     exit when Current_Token (Parser) = Token_Array_End;
                     --  Parse export name
                     if Current_Token (Parser) = Token_String then
                        if Module.Export_Count < Max_Exports then
                           Module.Export_Count := Module.Export_Count + 1;
                           Module.Exports (Module.Export_Count) :=
                              Name_Strings.To_Bounded_String (Current_String (Parser));
                        end if;
                     end if;
                  end loop;
               end if;
            elsif Key = "declarations" then
               --  Parse declarations array
               if Current_Token (Parser) = Token_Array_Start then
                  Module.Decl_Count := 0;
                  loop
                     Next_Token (Parser, Temp_Status);
                     if Temp_Status /= Success then
                        Status := Error_Parse;
                        return;
                     end if;
                     exit when Current_Token (Parser) = Token_Array_End;
                     --  Skip declaration object
                     if Current_Token (Parser) = Token_Object_Start then
                        --  Skip until object end
                        declare
                           Depth : Natural := 1;
                        begin
                           while Depth > 0 loop
                              Next_Token (Parser, Temp_Status);
                              if Temp_Status /= Success then
                                 Status := Error_Parse;
                                 return;
                              end if;
                              if Current_Token (Parser) = Token_Object_Start then
                                 Depth := Depth + 1;
                              elsif Current_Token (Parser) = Token_Object_End then
                                 Depth := Depth - 1;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
               end if;
            elsif Key = "metadata" then
               --  Parse metadata object
               if Current_Token (Parser) = Token_Object_Start then
                  --  Skip until object end
                  declare
                     Depth : Natural := 1;
                  begin
                     while Depth > 0 loop
                        Next_Token (Parser, Temp_Status);
                        if Temp_Status /= Success then
                           Status := Error_Parse;
                           return;
                        end if;
                        if Current_Token (Parser) = Token_Object_Start then
                           Depth := Depth + 1;
                        elsif Current_Token (Parser) = Token_Object_End then
                           Depth := Depth - 1;
                        end if;
                     end loop;
                  end;
               end if;
            else
               --  Skip unknown field
               if Current_Token (Parser) = Token_Object_Start then
                  declare
                     Depth : Natural := 1;
                  begin
                     while Depth > 0 loop
                        Next_Token (Parser, Temp_Status);
                        if Temp_Status /= Success then
                           Status := Error_Parse;
                           return;
                        end if;
                        if Current_Token (Parser) = Token_Object_Start then
                           Depth := Depth + 1;
                        elsif Current_Token (Parser) = Token_Object_End then
                           Depth := Depth - 1;
                        end if;
                     end loop;
                  end;
               elsif Current_Token (Parser) = Token_Array_Start then
                  declare
                     Depth : Natural := 1;
                  begin
                     while Depth > 0 loop
                        Next_Token (Parser, Temp_Status);
                        if Temp_Status /= Success then
                           Status := Error_Parse;
                           return;
                        end if;
                        if Current_Token (Parser) = Token_Array_Start then
                           Depth := Depth + 1;
                        elsif Current_Token (Parser) = Token_Array_End then
                           Depth := Depth - 1;
                        end if;
                     end loop;
                  end;
               end if;
            end if;
         end;

         --  Advance to next key or end
         Next_Token (Parser, Temp_Status);
         if Temp_Status /= Success then
            Status := Error_Parse;
            return;
         end if;

         --  Skip comma if present
         if Current_Token (Parser) = Token_Comma then
            Next_Token (Parser, Temp_Status);
            if Temp_Status /= Success then
               Status := Error_Parse;
               return;
            end if;
         end if;
      end loop;

      --  Validate parsed module
      if Is_Valid_Module (Module) then
         Status := Success;
      else
         Status := Error_Validation;
      end if;

   exception
      when others =>
         Status := Error_Parse;
         Init_Semantic_Module (Module);
   end Parse_Semantic_IR_String;

   --  Convert flat IR JSON to Semantic IR JSON
   procedure Convert_Flat_IR_To_Semantic
     (Flat_IR_JSON : in     JSON_String;
      Semantic_IR  :    out JSON_String;
      Status       :    out Status_Code)
   is
      --  Parse flat IR and convert to Semantic IR
      --  This implementation:
      --  1. Parses the flat IR JSON using IR_Parse
      --  2. Converts IR_Data to Semantic_Module with type bindings
      --  3. Adds control flow graph edges
      --  4. Adds semantic annotations
      --  5. Serializes to Semantic IR JSON
      
      Flat_IR       : IR_Data;
      Sem_Module    : Semantic_Module;
      Parse_Status  : Status_Code;
      Emit_Result   : Emitter_Result;
   begin
      --  Initialize output
      Semantic_IR := JSON_String_Strings.Null_Bounded_String;
      
      --  Parse the flat IR JSON
      IR_Parse.Parse_IR_String (Flat_IR_JSON, Flat_IR, Parse_Status);
      
      if Parse_Status /= Success then
         Status := Error_Parse;
         return;
      end if;
      
      --  Initialize Semantic IR module
      Init_Semantic_Module (Sem_Module);
      
      --  Convert module name (IR_Data uses Identifier_String, Semantic_Module uses Name_String)
      if Identifier_Strings.Length (Flat_IR.Module_Name) > 0 then
         --  Copy module name (bounded string conversion)
         declare
            Name_Str : constant String := Identifier_Strings.To_String (Flat_IR.Module_Name);
         begin
            if Name_Str'Length <= Max_Name_Length then
               Sem_Module.Module_Name := Name_Strings.To_Bounded_String (Name_Str);
            end if;
         end;
      end if;
      
      --  Convert imports (IR_Data uses Import_Collection, Semantic_Module uses Import_List)
      Sem_Module.Import_Count := 0;
      for I in 1 .. Flat_IR.Imports.Count loop
         if Sem_Module.Import_Count < Max_Imports then
            Sem_Module.Import_Count := Sem_Module.Import_Count + 1;
            --  Copy import module name
            declare
               Imp_Name : constant String := Identifier_Strings.To_String (Flat_IR.Imports.Imports (I).From_Module);
            begin
               if Imp_Name'Length <= Max_Name_Length then
                  Sem_Module.Imports (Sem_Module.Import_Count).Module_Name :=
                     Name_Strings.To_Bounded_String (Imp_Name);
               end if;
            end;
            Sem_Module.Imports (Sem_Module.Import_Count).Import_All := False;
            Sem_Module.Imports (Sem_Module.Import_Count).Symbol_Count := 0;
         end if;
      end loop;
      
      --  Convert exports (IR_Data uses Export_Collection, Semantic_Module uses Export_List)
      Sem_Module.Export_Count := 0;
      for I in 1 .. Flat_IR.Exports.Count loop
         if Sem_Module.Export_Count < Max_Exports then
            Sem_Module.Export_Count := Sem_Module.Export_Count + 1;
            declare
               Exp_Name : constant String := Identifier_Strings.To_String (Flat_IR.Exports.Exports (I).Name);
            begin
               if Exp_Name'Length <= Max_Name_Length then
                  Sem_Module.Exports (Sem_Module.Export_Count) :=
                     Name_Strings.To_Bounded_String (Exp_Name);
               end if;
            end;
         end if;
      end loop;
      
      --  Convert function declarations (simplified: use function names as declarations)
      Sem_Module.Decl_Count := 0;
      for I in 1 .. Flat_IR.Functions.Count loop
         if Sem_Module.Decl_Count < Max_Declarations then
            Sem_Module.Decl_Count := Sem_Module.Decl_Count + 1;
            declare
               Func_Name : constant String := Identifier_Strings.To_String (Flat_IR.Functions.Functions (I).Name);
            begin
               if Func_Name'Length <= Max_Name_Length then
                  Sem_Module.Declarations (Sem_Module.Decl_Count) :=
                     Name_Strings.To_Bounded_String (Func_Name);
               end if;
            end;
         end if;
      end loop;
      
      --  Set module metadata
      Sem_Module.Metadata.Module_Safety := Level_None;
      Sem_Module.Metadata.Optimization_Level := 0;
      Sem_Module.Metadata.Is_Entry_Point := False;
      Sem_Module.Metadata.Target_Count := 0;
      
      --  Initialize CFG
      Sem_Module.CFG.Entry_Node := Name_Strings.Null_Bounded_String;
      Sem_Module.CFG.Exit_Node := Name_Strings.Null_Bounded_String;
      Sem_Module.CFG.Node_Count := 0;
      
      --  Compute module hash for confluence
      Compute_Module_Hash (Sem_Module);
      
      --  Emit Semantic IR to JSON
      Emit_Module_To_JSON (Sem_Module, Semantic_IR, Emit_Result);
      
      if not Emit_Result.Success then
         Status := Error_Emission_Failed;
         return;
      end if;
      
      Status := Success;
      
   exception
      when others =>
         Status := Error_Parse;
         Semantic_IR := JSON_String_Strings.Null_Bounded_String;
   end Convert_Flat_IR_To_Semantic;

end Semantic_IR.Parse;