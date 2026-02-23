--  source_extract - Minimal source -> extraction.json (C/C++)
--  Phase 0: Source extraction
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with STUNIR_Types;
use STUNIR_Types;

package Source_Extract is

   procedure Extract_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Language    : in     Identifier_String;
      Status      :    out Status_Code);

end Source_Extract;
