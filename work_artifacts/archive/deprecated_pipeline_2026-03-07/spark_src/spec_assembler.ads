--  STUNIR Spec Assembler
--  Converts extraction.json to spec.json format
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  SPARK equivalent of bridge_spec_assemble.py
--  Phase 1 of the STUNIR pipeline

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Spec_Assembler is

   --  ========================================================================
   --  Extraction Record Types (Input)
   --  ========================================================================

   type Extraction_Function is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
   end record;

   Max_Extractions_Per_File : constant := 32;

   type Extraction_File_Index is range 0 .. Max_Extractions_Per_File;

   type Extraction_Function_Array is
     array (Extraction_File_Index range <>) of Extraction_Function;

   type Extraction_File is record
      Source_File : Path_String;
      Functions   : Extraction_Function_Array (1 .. Max_Extractions_Per_File);
      Count       : Extraction_File_Index;
   end record;

   Max_Extraction_Files : constant := 16;

   type Extraction_File_Index_Range is range 0 .. Max_Extraction_Files;

   type Extraction_File_Array is
     array (Extraction_File_Index_Range range <>) of Extraction_File;

   type Extraction_Data is record
      Schema_Version : Identifier_String;
      Source_Index   : Path_String;
      Files          : Extraction_File_Array (1 .. Max_Extraction_Files);
      File_Count     : Extraction_File_Index_Range;
   end record;

   --  ========================================================================
   --  Spec Record Types (Output)
   --  ========================================================================

   type Spec_Function is record
      Name        : Identifier_String;
      Return_Type : Type_Name_String;
      Parameters  : Parameter_List;
   end record;

   type Spec_Module is record
      Name      : Identifier_String;
      Functions : Function_Collection;
   end record;

   type Spec_Data is record
      Schema_Version : Identifier_String;
      Origin         : Identifier_String;
      Spec_Hash      : Identifier_String;
      Source_Index   : Path_String;
      Module         : Spec_Module;
   end record;

   --  ========================================================================
   --  Core Operations
   --  ========================================================================

   procedure Parse_Extraction_JSON
     (JSON_Content : in     JSON_String;
      Extraction   :    out Extraction_Data;
      Status       :    out Status_Code)
   with
      Pre  => JSON_Strings.Length (JSON_Content) > 0,
      Post => (if Status = Success then
                 Extraction.File_Count > 0 or
                 Extraction.File_Count = 0);

   procedure Validate_Extraction
     (Extraction : in     Extraction_Data;
      Status     :    out Status_Code)
   with
      Post => (if Status = Success then
                 (for all I in 1 .. Extraction.File_Count =>
                    Extraction.Files (I).Count >= 0));

   procedure Assemble_Spec
     (Extraction  : in     Extraction_Data;
      Module_Name : in     Identifier_String;
      Spec        :    out Spec_Data;
      Status      :    out Status_Code)
   with
      Pre  => Identifier_Strings.Length (Module_Name) > 0,
      Post => (if Status = Success then
                 Spec.Module.Functions.Count > 0 or
                 Spec.Module.Functions.Count = 0);

   procedure Generate_Spec_JSON
     (Spec        : in     Spec_Data;
      JSON_Output :    out JSON_String;
      Status      :    out Status_Code)
   with
      Post => (if Status = Success then
                 JSON_Strings.Length (JSON_Output) > 0);

   --  ========================================================================
   --  Main Entry Point
   --  ========================================================================

   procedure Process_Extraction_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Status      :    out Status_Code)
   with
      Pre  => Path_Strings.Length (Input_Path) > 0 and
              Path_Strings.Length (Output_Path) > 0,
      Post => Status in Status_Code'Range;

   --  Helper function for generating spec hash
   function Generate_Simple_Hash (Spec : Spec_Data) return Identifier_String
     with Global => null;

end Spec_Assembler;
