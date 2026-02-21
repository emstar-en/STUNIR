--  STUNIR External Tool Interface Specification
--  Safe external tool execution and discovery
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package External_Tool is

   Max_Tool_Name_Length : constant := 64;
   Max_Tool_Path_Length : constant := 512;
   Max_Version_Length   : constant := 64;
   Max_Output_Length    : constant := 65536;
   Max_Command_Length   : constant := 1024;
   Max_Arguments        : constant := 32;
   Max_Argument_Length  : constant := 256;
   Max_Tools            : constant := 64;
   Max_Timeout_MS       : constant := 300000;

   type Tool_Status is (
      Success,
      Tool_Not_Found,
      Execution_Failed,
      Timeout_Exceeded,
      Output_Overflow,
      Invalid_Arguments,
      Permission_Denied,
      Unknown_Error
   );

   subtype Tool_Name_Index is Positive range 1 .. Max_Tool_Name_Length;
   subtype Tool_Name_Length is Natural range 0 .. Max_Tool_Name_Length;
   subtype Tool_Name_String is String (Tool_Name_Index);

   subtype Tool_Path_Index is Positive range 1 .. Max_Tool_Path_Length;
   subtype Tool_Path_Length is Natural range 0 .. Max_Tool_Path_Length;
   subtype Tool_Path_String is String (Tool_Path_Index);

   subtype Version_Index is Positive range 1 .. Max_Version_Length;
   subtype Version_Len_Type is Natural range 0 .. Max_Version_Length;
   subtype Version_String is String (Version_Index);

   subtype Output_Index is Positive range 1 .. Max_Output_Length;
   subtype Output_Len_Type is Natural range 0 .. Max_Output_Length;
   subtype Output_Buffer is String (Output_Index);

   subtype Argument_Index is Positive range 1 .. Max_Argument_Length;
   subtype Argument_Len_Type is Natural range 0 .. Max_Argument_Length;
   subtype Argument_String is String (Argument_Index);

   subtype Exit_Code_Range is Integer range -128 .. 255;

   type Tool_Item is record
      Name        : Tool_Name_String;
      Name_Len    : Tool_Name_Length;
      Path        : Tool_Path_String;
      Path_Len    : Tool_Path_Length;
      Version     : Version_String;
      Version_Len : Version_Len_Type;
      Available   : Boolean;
      Verified    : Boolean;
   end record;

   Null_Tool_Item : constant Tool_Item := (
      Name        => (others => ' '),
      Name_Len    => 0,
      Path        => (others => ' '),
      Path_Len    => 0,
      Version     => (others => ' '),
      Version_Len => 0,
      Available   => False,
      Verified    => False
   );

   subtype Tool_Index is Positive range 1 .. Max_Tools;
   subtype Tool_Count is Natural range 0 .. Max_Tools;
   type Tool_Array is array (Tool_Index) of Tool_Item;

   type Tool_Registry is record
      Tools       : Tool_Array;
      Count       : Tool_Count;
      Initialized : Boolean;
   end record;

   Null_Tool_Registry : constant Tool_Registry := (
      Tools       => (others => Null_Tool_Item),
      Count       => 0,
      Initialized => False
   );

   type Argument_Item is record
      Value     : Argument_String;
      Value_Len : Argument_Len_Type;
   end record;

   Null_Argument_Item : constant Argument_Item := (
      Value     => (others => ' '),
      Value_Len => 0
   );

   subtype Argument_Array_Index is Positive range 1 .. Max_Arguments;
   subtype Argument_Count is Natural range 0 .. Max_Arguments;
   type Argument_Array is array (Argument_Array_Index) of Argument_Item;

   type Command_Result is record
      Exit_Code   : Exit_Code_Range;
      Output      : Output_Buffer;
      Output_Len  : Output_Len_Type;
      Error_Out   : Output_Buffer;
      Error_Len   : Output_Len_Type;
      Timed_Out   : Boolean;
      Success     : Boolean;
   end record;

   Null_Command_Result : constant Command_Result := (
      Exit_Code   => 0,
      Output      => (others => ' '),
      Output_Len  => 0,
      Error_Out   => (others => ' '),
      Error_Len   => 0,
      Timed_Out   => False,
      Success     => False
   );

   procedure Initialize_Registry (Registry : out Tool_Registry)
   with Post => Registry.Initialized and Registry.Count = 0;

   procedure Discover_Tool
     (Name   : String;
      Item   : out Tool_Item;
      Status : out Tool_Status)
   with Pre  => Name'Length > 0 and Name'Length <= Max_Tool_Name_Length,
        Post => (if Status = Success then Item.Available);

   procedure Register_Tool
     (Registry : in out Tool_Registry;
      Item     : Tool_Item;
      Status   : out Tool_Status)
   with Pre  => Registry.Initialized and
                Registry.Count < Max_Tools and Item.Available,
        Post => (if Status = Success then
                  Registry.Count = Registry.Count'Old + 1);

   function Find_Tool
     (Registry : Tool_Registry;
      Name     : String) return Tool_Count
   with Pre => Registry.Initialized and
               Name'Length > 0 and Name'Length <= Max_Tool_Name_Length;

   procedure Execute_Command
     (Command    : String;
      Args       : Argument_Array;
      Arg_Count  : Argument_Count;
      Timeout_MS : Positive;
      Result     : out Command_Result;
      Status     : out Tool_Status)
   with Pre  => Command'Length > 0 and Command'Length <= Max_Command_Length and
                Timeout_MS <= Max_Timeout_MS,
        Post => (if Status = Success then Result.Success);

   procedure Get_Tool_Version
     (Item   : in out Tool_Item;
      Status : out Tool_Status)
   with Pre => Item.Available;

   function Status_Message (Status : Tool_Status) return String;

   function Tool_Exists (Item : Tool_Item) return Boolean
   with Post => Tool_Exists'Result = Item.Available;

   procedure Build_Arguments
     (Args      : out Argument_Array;
      Arg_Count : out Argument_Count;
      Arg1      : String := "";
      Arg2      : String := "";
      Arg3      : String := "";
      Arg4      : String := "")
   with Pre => Arg1'Length <= Max_Argument_Length and
               Arg2'Length <= Max_Argument_Length and
               Arg3'Length <= Max_Argument_Length and
               Arg4'Length <= Max_Argument_Length;

end External_Tool;
