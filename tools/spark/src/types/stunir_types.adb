--  STUNIR Types Package Body
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body STUNIR_Types is

   function Status_Code_Image (Status : Status_Code) return String is
   begin
      case Status is
         when Success                 => return "Success";
         when Error_File_Not_Found    => return "Error: File not found";
         when Error_File_Read         => return "Error: File read failed";
         when Error_File_Write        => return "Error: File write failed";
         when Error_Invalid_JSON      => return "Error: Invalid JSON";
         when Error_Invalid_Schema    => return "Error: Invalid schema";
         when Error_Invalid_Syntax    => return "Error: Invalid syntax";
         when Error_Buffer_Overflow   => return "Error: Buffer overflow";
         when Error_Unsupported_Type  => return "Error: Unsupported type";
         when Error_Parse_Error       => return "Error: Parse error";
         when Error_Validation_Failed => return "Error: Validation failed";
         when Error_Conversion_Failed => return "Error: Conversion failed";
         when Error_Emission_Failed   => return "Error: Code emission failed";
         when Error_Not_Implemented   => return "Error: Not implemented";
         when Error_Invalid_Format    => return "Error: Invalid format";
         when Error_Empty_Extraction  => return "Error: Empty extraction";
         when Error_Too_Large         => return "Error: Too large";
         when Error_Parse             => return "Error: Parse";
         when Error_File_IO           => return "Error: File I/O failed";
         when Error_Invalid_Input     => return "Error: Invalid input";
      end case;
   end Status_Code_Image;

   function Is_Success (Status : Status_Code) return Boolean is
   begin
      return Status = Success;
   end Is_Success;

   function Is_Error (Status : Status_Code) return Boolean is
   begin
      return Status /= Success;
   end Is_Error;

end STUNIR_Types;
