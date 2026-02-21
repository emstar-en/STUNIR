--  STUNIR Report Formatter Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Report_Types; use Report_Types;

package Report_Formatter is

   procedure Initialize_Report
     (Data   : out Report_Data;
      Schema : String;
      Title  : String;
      Format : Report_Format;
      Status : out Report_Status)
   with Pre => Schema'Length > 0 and Schema'Length <= Max_Schema_Length and
               Title'Length > 0 and Title'Length <= Max_Title_Length,
        Post => (if Status = Success then Data.Is_Valid else not Data.Is_Valid);

   procedure Add_String_Item
     (Data   : in out Report_Data;
      Key    : String;
      Value  : String;
      Status : out Report_Status)
   with Pre => Data.Is_Valid and
               Key'Length > 0 and Key'Length <= Max_Key_Length and
               Value'Length <= Max_Value_Length and
               Data.Flat_Count < Max_Report_Items,
        Post => (if Status = Success then Data.Flat_Count = Data.Flat_Count'Old + 1);

   procedure Add_Integer_Item
     (Data   : in Out Report_Data;
      Key    : String;
      Value  : Integer;
      Status : out Report_Status)
   with Pre => Data.Is_Valid and
               Key'Length > 0 and Key'Length <= Max_Key_Length and
               Data.Flat_Count < Max_Report_Items,
        Post => (if Status = Success then Data.Flat_Count = Data.Flat_Count'Old + 1);

   procedure Add_Boolean_Item
     (Data   : in Out Report_Data;
      Key    : String;
      Value  : Boolean;
      Status : out Report_Status)
   with Pre => Data.Is_Valid and
               Key'Length > 0 and Key'Length <= Max_Key_Length and
               Data.Flat_Count < Max_Report_Items,
        Post => (if Status = Success then Data.Flat_Count = Data.Flat_Count'Old + 1);

   procedure Add_Section
     (Data         : in Out Report_Data;
      Section_Name : String;
      Status       : out Report_Status)
   with Pre => Data.Is_Valid and
               Section_Name'Length > 0 and
               Section_Name'Length <= Max_Section_Name and
               Data.Section_Total < Max_Sections,
        Post => (if Status = Success then
                  Data.Section_Total = Data.Section_Total'Old + 1 and
                  Data.Has_Sections);

   procedure Generate_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   with Pre => Data.Is_Valid and Has_Items (Data),
        Post => (if Status = Success then Length > 0);

   procedure Generate_Text_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   with Pre => Data.Is_Valid,
        Post => (if Status = Success then Length > 0);

   procedure Generate_JSON_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   with Pre => Data.Is_Valid,
        Post => (if Status = Success then Length > 0);

   procedure Generate_HTML_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   with Pre => Data.Is_Valid,
        Post => (if Status = Success then Length > 0);

   procedure Generate_XML_Report
     (Data   : Report_Data;
      Output : out Output_Buffer;
      Length : out Output_Length;
      Status : out Report_Status)
   with Pre => Data.Is_Valid,
        Post => (if Status = Success then Length > 0);

   procedure Clear_Report (Data : in Out Report_Data)
   with Post => Data.Flat_Count = 0 and Data.Section_Total = 0;

   function Total_Item_Count (Data : Report_Data) return Natural;

   procedure Sort_Items_By_Key
     (Items : in Out Item_Array;
      Count : Item_Count)
   with Pre => Count <= Max_Report_Items;

end Report_Formatter;
