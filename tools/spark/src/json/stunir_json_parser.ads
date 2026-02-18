--  STUNIR JSON Parser Package
--  Streaming JSON parser for SPARK with formal contracts
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides a streaming JSON parser suitable for formal
--  verification. It uses bounded strings and fixed-size buffers.

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package STUNIR_JSON_Parser is

   pragma Pure;

   --  ========================================================================
   --  JSON Token Types
   --  ========================================================================

   type Token_Kind is (
      Token_EOF,
      Token_Object_Start,   --  {
      Token_Object_End,     --  }
      Token_Array_Start,    --  [
      Token_Array_End,      --  ]
      Token_Colon,          --  :
      Token_Comma,          --  ,
      Token_String,
      Token_Number,
      Token_True,
      Token_False,
      Token_Null
   );

   --  ========================================================================
   --  Parser State
   --  ========================================================================

   Max_Nesting_Depth : constant := 64;

   type Nesting_Kind is (Nest_Object, Nest_Array);

   type Nesting_Stack is array (1 .. Max_Nesting_Depth) of Nesting_Kind;

   type Parser_State is record
      Input        : JSON_String;
      Position     : Positive range 1 .. Max_JSON_Length + 1;
      Line         : Positive range 1 .. Max_JSON_Length;
      Column       : Positive range 1 .. Max_JSON_Length;
      Nesting      : Nesting_Stack;
      Nesting_Level : Natural range 0 .. Max_Nesting_Depth;
      Current_Token : Token_Kind;
      Token_Value  : JSON_String;
   end record;

   --  ========================================================================
   --  Parser Operations
   --  ========================================================================

   procedure Initialize_Parser
     (State  : out Parser_State;
      Input  : in  JSON_String;
      Status : out Status_Code)
   with
      Pre  => JSON_Strings.Length (Input) > 0,
      Post => (if Status = Success then State.Position = 1);

   procedure Next_Token
     (State  : in out Parser_State;
      Status : out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => (if Status = Success then
                 State.Current_Token in Token_Kind'Range);

   function Current_Token (State : Parser_State) return Token_Kind
   with
      Post => Current_Token'Result in Token_Kind'Range;

   function Token_String_Value (State : Parser_State) return JSON_String;

   procedure Expect_Token
     (State      : in out Parser_State;
      Expected   : in     Token_Kind;
      Status     :    out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => (if Status = Success then State.Current_Token = Expected);

   --  ========================================================================
   --  Object Navigation
   --  ========================================================================

   procedure Skip_Value
     (State  : in out Parser_State;
      Status : out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => State.Position >= State.Position'Old;

   procedure Parse_String_Member
     (State       : in out Parser_State;
      Member_Name :    out Identifier_String;
      Member_Value:    out JSON_String;
      Status      :    out Status_Code)
   with
      Pre  => State.Position <= Max_JSON_Length,
      Post => (if Status = Success then
                 Identifier_Strings.Length (Member_Name) > 0);

   --  ========================================================================
   --  Utility Functions
   --  ========================================================================

   function Is_At_End (State : Parser_State) return Boolean;

   function Get_Position_Line (State : Parser_State) return Positive;

   function Get_Position_Column (State : Parser_State) return Positive;

end STUNIR_JSON_Parser;
