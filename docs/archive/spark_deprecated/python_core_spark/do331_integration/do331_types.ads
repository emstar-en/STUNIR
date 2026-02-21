--  STUNIR DO-331 Integration Types
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package DO331_Types is

   Max_Model_Name_Length : constant := 128;
   Max_Model_Path_Length : constant := 512;
   Max_Model_Count       : constant := 256;
   Max_Trace_Links       : constant := 1024;
   Max_Coverage_Items    : constant := 512;
   Max_Element_ID_Length : constant := 64;

   type DAL_Level is (DAL_A, DAL_B, DAL_C, DAL_D, DAL_E);

   type Model_Kind is (
      Block_Model,
      Activity_Model,
      StateMachine_Model,
      Sequence_Model,
      Requirement_Model,
      Package_Model,
      Class_Model,
      Interface_Model
   );

   subtype Model_Name_Index is Positive range 1 .. Max_Model_Name_Length;
   subtype Model_Name_Length is Natural range 0 .. Max_Model_Name_Length;
   subtype Model_Name_String is String (Model_Name_Index);

   subtype Model_Path_Index is Positive range 1 .. Max_Model_Path_Length;
   subtype Model_Path_Length is Natural range 0 .. Max_Model_Path_Length;
   subtype Model_Path_String is String (Model_Path_Index);

   subtype Element_ID_Index is Positive range 1 .. Max_Element_ID_Length;
   subtype Element_ID_Length is Natural range 0 .. Max_Element_ID_Length;
   subtype Element_ID_String is String (Element_ID_Index);

   subtype Percentage_Type is Float range 0.0 .. 100.0;

   type Coverage_Kind is (
      Statement_Coverage, Branch_Coverage, MCDC_Coverage,
      State_Coverage, Transition_Coverage, Path_Coverage
   );

   type Coverage_Item is record
      Element_ID  : Element_ID_String;
      Element_Len : Element_ID_Length;
      Kind        : Coverage_Kind;
      Covered     : Boolean;
      Hit_Count   : Natural;
   end record;

   Null_Coverage_Item : constant Coverage_Item := (
      Element_ID  => (others => ' '),
      Element_Len => 0,
      Kind        => Statement_Coverage,
      Covered     => False,
      Hit_Count   => 0
   );

   type Trace_Direction is (Forward, Backward, Bidirectional);

   type Trace_Link is record
      Source_ID   : Element_ID_String;
      Source_Len  : Element_ID_Length;
      Target_ID   : Element_ID_String;
      Target_Len  : Element_ID_Length;
      Direction   : Trace_Direction;
      Is_Valid    : Boolean;
   end record;

   Null_Trace_Link : constant Trace_Link := (
      Source_ID  => (others => ' '),
      Source_Len => 0,
      Target_ID  => (others => ' '),
      Target_Len => 0,
      Direction  => Forward,
      Is_Valid   => False
   );

   type Model_Item is record
      Name          : Model_Name_String;
      Name_Len      : Model_Name_Length;
      Path          : Model_Path_String;
      Path_Len      : Model_Path_Length;
      Kind          : Model_Kind;
      Element_Count : Natural;
      Is_Valid      : Boolean;
   end record;

   Null_Model_Item : constant Model_Item := (
      Name          => (others => ' '),
      Name_Len      => 0,
      Path          => (others => ' '),
      Path_Len      => 0,
      Kind          => Block_Model,
      Element_Count => 0,
      Is_Valid      => False
   );

   subtype Model_Index is Positive range 1 .. Max_Model_Count;
   subtype Model_Count is Natural range 0 .. Max_Model_Count;
   type Model_Array is array (Model_Index) of Model_Item;

   subtype Trace_Index is Positive range 1 .. Max_Trace_Links;
   subtype Trace_Count is Natural range 0 .. Max_Trace_Links;
   type Trace_Array is array (Trace_Index) of Trace_Link;

   subtype Coverage_Index is Positive range 1 .. Max_Coverage_Items;
   subtype Coverage_Count is Natural range 0 .. Max_Coverage_Items;
   type Coverage_Array is array (Coverage_Index) of Coverage_Item;

   type DO331_Result is record
      Models         : Model_Array;
      Model_Total    : Model_Count;
      Coverage_Items : Coverage_Array;
      Coverage_Total : Coverage_Count;
      Coverage_Pct   : Percentage_Type;
      Trace_Links    : Trace_Array;
      Trace_Total    : Trace_Count;
      DAL            : DAL_Level;
      Success        : Boolean;
      Is_Complete    : Boolean;
   end record;

   Null_DO331_Result : constant DO331_Result := (
      Models         => (others => Null_Model_Item),
      Model_Total    => 0,
      Coverage_Items => (others => Null_Coverage_Item),
      Coverage_Total => 0,
      Coverage_Pct   => 0.0,
      Trace_Links    => (others => Null_Trace_Link),
      Trace_Total    => 0,
      DAL            => DAL_E,
      Success        => False,
      Is_Complete    => False
   );

   type DO331_Status is (
      Success, Model_Not_Found, Invalid_Model, Transform_Failed,
      Coverage_Incomplete, Trace_Missing, IO_Error
   );

   function Status_Message (Status : DO331_Status) return String;
   function DAL_Name (Level : DAL_Level) return String;
   function Model_Kind_Name (Kind : Model_Kind) return String;

end DO331_Types;
