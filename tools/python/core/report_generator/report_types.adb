--  STUNIR Report Generator - Type Definitions Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Report_Types is

   function Format_Name (Format : Report_Format) return String is
   begin
      case Format is
         when Text_Format => return "text";
         when JSON_Format => return "json";
         when HTML_Format => return "html";
         when XML_Format  => return "xml";
      end case;
   end Format_Name;

   function Status_Message (Status : Report_Status) return String is
   begin
      case Status is
         when Success               => return "Operation completed successfully";
         when Invalid_Format        => return "Invalid report format specified";
         when Buffer_Overflow       => return "Output buffer overflow";
         when Invalid_Data          => return "Invalid report data";
         when Missing_Required_Field=> return "Required field is missing";
         when Encoding_Error        => return "Encoding error occurred";
         when IO_Error              => return "I/O error occurred";
      end case;
   end Status_Message;

   function Is_String_Item (Item : Report_Item) return Boolean is
   begin
      return Item.Is_Valid and Item.Kind = String_Value;
   end Is_String_Item;

   function Has_Items (Data : Report_Data) return Boolean is
   begin
      return Data.Flat_Count > 0 or
             (Data.Has_Sections and Data.Section_Total > 0);
   end Has_Items;

end Report_Types;
