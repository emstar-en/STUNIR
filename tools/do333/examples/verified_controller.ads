--  STUNIR DO-333 Example: Formally Verified Controller
--  Demonstrates formal specification patterns
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package Verified_Controller is

   --  ============================================================
   --  Type Definitions with Predicates
   --  ============================================================

   --  Altitude type with range constraint
   subtype Altitude_Type is Integer range 0 .. 50_000;
   --  Altitude in feet, 0 to 50,000 ft

   Max_Altitude : constant Altitude_Type := 45_000;
   --  Maximum operational altitude

   --  Rate of change with bounds
   subtype Rate_Type is Integer range -5_000 .. 5_000;
   --  Feet per minute, +/- 5000 fpm

   Max_Rate : constant Rate_Type := 4_000;
   --  Maximum normal rate

   --  Controller mode
   type Controller_Mode is (Mode_Idle, Mode_Climbing, Mode_Descending, Mode_Level);

   --  ============================================================
   --  State Variables with Type Invariants
   --  ============================================================

   --  Current altitude (module state)
   Current_Altitude : Altitude_Type := 0
     with Part_Of => Controller_State;

   --  Target altitude
   Target_Altitude : Altitude_Type := 0
     with Part_Of => Controller_State;

   --  Current rate
   Current_Rate : Rate_Type := 0
     with Part_Of => Controller_State;

   --  Current mode
   Mode : Controller_Mode := Mode_Idle
     with Part_Of => Controller_State;

   --  Abstract state for SPARK
   Controller_State : Integer := 0
     with Ghost;

   --  ============================================================
   --  Ghost Functions for Specifications
   --  ============================================================

   --  Check if controller is in valid state
   function Is_Valid_State return Boolean is
     (Current_Altitude <= Max_Altitude and then
      abs (Current_Rate) <= Max_Rate)
   with Ghost;

   --  Check if altitude within bounds
   function Altitude_In_Bounds (Alt : Altitude_Type) return Boolean is
     (Alt <= Max_Altitude)
   with Ghost;

   --  ============================================================
   --  Operations with Formal Contracts
   --  ============================================================

   --  Initialize controller to safe state
   procedure Initialize
   with
      Global => (Output => (Current_Altitude, Target_Altitude,
                           Current_Rate, Mode)),
      Post   => Current_Altitude = 0 and then
                Target_Altitude = 0 and then
                Current_Rate = 0 and then
                Mode = Mode_Idle;

   --  Set target altitude
   procedure Set_Target (Target : Altitude_Type)
   with
      Global => (In_Out => Target_Altitude,
                 Input  => Current_Altitude,
                 Output => Mode),
      Pre    => Target <= Max_Altitude,
      Post   => Target_Altitude = Target and then
                (if Target > Current_Altitude then Mode = Mode_Climbing
                 elsif Target < Current_Altitude then Mode = Mode_Descending
                 else Mode = Mode_Level);

   --  Update altitude based on rate
   procedure Update_Altitude (Delta_Time : Positive)
   with
      Global => (In_Out => (Current_Altitude, Current_Rate),
                 Input  => (Target_Altitude, Mode)),
      Pre    => Delta_Time <= 60 and then  --  Max 60 seconds
                Is_Valid_State,
      Post   => Current_Altitude <= Max_Altitude and then
                Is_Valid_State;

   --  Emergency stop - immediately level off
   procedure Emergency_Stop
   with
      Global => (Output => (Current_Rate, Mode)),
      Post   => Current_Rate = 0 and then
                Mode = Mode_Level;

   --  Get current altitude (query)
   function Get_Altitude return Altitude_Type
   with
      Global => (Input => Current_Altitude);

   --  Get current mode (query)
   function Get_Mode return Controller_Mode
   with
      Global => (Input => Mode);

   --  Check if at target
   function At_Target return Boolean
   with
      Global => (Input => (Current_Altitude, Target_Altitude)),
      Post   => At_Target'Result = (Current_Altitude = Target_Altitude);

end Verified_Controller;
