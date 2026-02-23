--  spark_extract - Minimal SPARK source -> extraction.json
--  Phase 0: Source extraction (SPARK)
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with STUNIR_Types;

package Spark_Extract is
   use STUNIR_Types;

   procedure Extract_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Language    : in     Identifier_String;
      Status      :    out Status_Code);

  function Last_Error return String;

end Spark_Extract;
