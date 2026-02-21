--  STUNIR Report Generator - Type Definitions
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Report_Types is

   Max_Key_Length      : constant := 64;
   Max_Value_Length    : constant := 1024;
   Max_Report_Items    : constant := 128;
   Max_Output_Length   : constant := 131072;
   Max_Schema_Length   : constant := 64;
   Max_Title_Length    : constant := 128;
   Max_Section_Name    : constant := 64;
   Max_Sections        : constant := 32;

   type Report_Format is (Text_Format, JSON_Format, HTML_Format, XML_Format);

   type Report_Status is (
      Success, Invalid_Format, Buffer_Overflow, Invalid_Data,
      Missing_Required_Field, Encoding_Error, IO_Error
   );

   subtype Key_Index is Positive range 1 .. Max_Key_Length;
   subtype Key_Length is Natural range 0 .. Max_Key_Length;
   subtype Key_String is String (Key_Index);

   subtype Value_Index is Positive range 1 .. Max_Value_Length;
   subtype Value_Length is Natural range 0 .. Max_Value_Length;
   subtype Value_String is String (Value_Index);

   type Value_Kind is (String_Value, Integer_Value, Float_Value, Boolean_Value, Null_Value);

   type Report_Item is record
      Key        : Key_String;
      Key_Len    : Key_Length;
      Value      : Value_String;
      Value_Len  : Value_Length;
      Kind       : Value_Kind;
      Int_Val    : Integer;
      Float_Val  : Float;
      Bool_Val   : Boolean;
      Is_Valid   : Boolean;
   end record;

   Null_Report_Item : constant Report_Item := (
      Key       => (others => ' '),
      Key_Len   => 0,
      Value     => (others => ' '),
      Value_Len => 0,
      Kind      => String_Value,
      Int_Val   => 0,
      Float_Val => 0.0,
      Bool_Val  => False,
      Is_Valid  => False
   );

   subtype Item_Index is Positive range 1 .. Max_Report_Items;
   subtype Item_Count is Natural range 0 .. Max_Report_Items;
   type Item_Array is array (Item_Index) of Report_Item;

   subtype Section_Name_Index is Positive range 1 .. Max_Section_Name;
   subtype Section_Name_Length is Natural range 0 .. Max_Section_Name;
   subtype Section_Name_String is String (Section_Name_Index);

   type Report_Section is record
      Name        : Section_Name_String;
      Name_Len    : Section_Name_Length;
      Items       : Item_Array;
      Item_Total  : Item_Count;
      Is_Valid    : Boolean;
   end record;

   Null_Report_Section : constant Report_Section := (
      Name       => (others => ' '),
      Name_Len   => 0,
      Items      => (others => Null_Report_Item),
      Item_Total => 0,
      Is_Valid   => False
   );

   subtype Section_Index is Positive range 1 .. Max_Sections;
   subtype Section_Count is Natural range 0 .. Max_Sections;
   type Section_Array is array (Section_Index) of Report_Section;

   subtype Schema_Index is Positive range 1 .. Max_Schema_Length;
   subtype Schema_Length is Natural range 0 .. Max_Schema_Length;
   subtype Schema_String is String (Schema_Index);

   subtype Title_Index is Positive range 1 .. Max_Title_Length;
   subtype Title_Length is Natural range 0 .. Max_Title_Length;
   subtype Title_String is String (Title_Index);

   subtype Output_Index is Positive range 1 .. Max_Output_Length;
   subtype Output_Length is Natural range 0 .. Max_Output_Length;
   subtype Output_Buffer is String (Output_Index);

   type Report_Data is record
      Schema        : Schema_String;
      Schema_Len    : Schema_Length;
      Title         : Title_String;
      Title_Len     : Title_Length;
      Format        : Report_Format;
      Flat_Items    : Item_Array;
      Flat_Count    : Item_Count;
      Sections      : Section_Array;
      Section_Total : Section_Count;
      Is_Valid      : Boolean;
      Has_Sections  : Boolean;
   end record;

   Null_Report_Data : constant Report_Data := (
      Schema        => (others => ' '),
      Schema_Len    => 0,
      Title         => (others => ' '),
      Title_Len     => 0,
      Format        => Text_Format,
      Flat_Items    => (others => Null_Report_Item),
      Flat_Count    => 0,
      Sections      => (others => Null_Report_Section),
      Section_Total => 0,
      Is_Valid      => False,
      Has_Sections  => False
   );

   function Format_Name (Format : Report_Format) return String;
   function Status_Message (Status : Report_Status) return String;
   function Is_String_Item (Item : Report_Item) return Boolean
   with Post => Is_String_Item'Result = (Item.Is_Valid and Item.Kind = String_Value);
   function Has_Items (Data : Report_Data) return Boolean;

end Report_Types;
