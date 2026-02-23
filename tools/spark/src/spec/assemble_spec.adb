--  Assemble Spec Micro-Tool Body
--  Assembles spec JSON from extraction data
--  Phase: 1 (Spec)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Extract_Parse;
with Spec_Parse;
with Ada.Text_IO;
use Ada.Text_IO;

package body Assemble_Spec is

   procedure Assemble_From_Extract
     (Extract : in     Extract_Parse.Extract_Data;
      Spec    :    out Spec_Parse.Spec_Data;
      Status  :    out Status_Code)
   is
   begin
      Status := Success;
      
      Spec.Schema_Version := Identifier_Strings.To_Bounded_String ("stunir_spec_v1");
      Spec.Module_Name := Extract.Module_Name;
      Spec.Functions.Count := Extract.Functions.Count;
      
      for I in Function_Index range 1 .. Extract.Functions.Count loop
         Spec.Functions.Functions (I).Name := Extract.Functions.Functions (I).Name;
         Spec.Functions.Functions (I).Return_Type := Extract.Functions.Functions (I).Return_Type;
         Spec.Functions.Functions (I).Parameters := Extract.Functions.Functions (I).Parameters;
      end loop;
      
      for I in Function_Index range Extract.Functions.Count + 1 .. Max_Functions loop
         Spec.Functions.Functions (I).Name := Identifier_Strings.Null_Bounded_String;
         Spec.Functions.Functions (I).Return_Type := Type_Name_Strings.Null_Bounded_String;
         Spec.Functions.Functions (I).Parameters.Count := 0;
      end loop;
   end Assemble_From_Extract;

   procedure Assemble_Spec_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Status      :    out Status_Code)
   is
      pragma SPARK_Mode (Off);
      
      Extract : Extract_Parse.Extract_Data;
      Spec    : Spec_Parse.Spec_Data;
      Output  : File_Type;
   begin
      Extract_Parse.Parse_Extract_File (Input_Path, Extract, Status);
      if Status /= Success then
         return;
      end if;
      
      Assemble_From_Extract (Extract, Spec, Status);
      if Status /= Success then
         return;
      end if;
      
      Create (Output, Out_File, Path_Strings.To_String (Output_Path));
      Put_Line (Output, "{");
      Put_Line (Output, "  ""schema"": ""stunir_spec_v1"",");
      Put_Line (Output, "  ""module_name"": """ & Identifier_Strings.To_String (Spec.Module_Name) & """,");
      Put_Line (Output, "  ""functions"": [");
      
      for I in Function_Index range 1 .. Spec.Functions.Count loop
         Put_Line (Output, "    {");
         Put_Line (Output, "      ""name"": """ & Identifier_Strings.To_String (Spec.Functions.Functions (I).Name) & """,");
         Put_Line (Output, "      ""return_type"": """ & Type_Name_Strings.To_String (Spec.Functions.Functions (I).Return_Type) & """,");
         
         --  Emit args array
         Put (Output, "      ""args"": [");
         for J in Parameter_Index range 1 .. Spec.Functions.Functions (I).Parameters.Count loop
            Put (Output, "{""name"": """ & 
               Identifier_Strings.To_String (Spec.Functions.Functions (I).Parameters.Params (J).Name) & 
               """, ""type"": """ & 
               Type_Name_Strings.To_String (Spec.Functions.Functions (I).Parameters.Params (J).Param_Type) & """}");
            if J < Spec.Functions.Functions (I).Parameters.Count then
               Put (Output, ", ");
            end if;
         end loop;
         Put_Line (Output, "]");
         
         if I < Spec.Functions.Count then
            Put_Line (Output, "    },");
         else
            Put_Line (Output, "    }");
         end if;
      end loop;
      
      Put_Line (Output, "  ]");
      Put_Line (Output, "}");
      Close (Output);
      
      Status := Success;
   exception
      when others =>
         Status := Error_File_IO;
   end Assemble_Spec_File;

end Assemble_Spec;
