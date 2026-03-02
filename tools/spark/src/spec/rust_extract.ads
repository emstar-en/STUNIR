--  rust_extract - Minimal Rust source -> extraction.json
--  Phase 0: Source extraction (Rust)
--  SPARK_Mode: Off (file I/O + parsing)
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (Off);

with STUNIR_Types;
use STUNIR_Types;

package Rust_Extract is

   procedure Extract_File
     (Input_Path  : in     Path_String;
      Output_Path : in     Path_String;
      Module_Name : in     Identifier_String;
      Language    : in     Identifier_String;
      Status      :    out Status_Code);

end Rust_Extract;
