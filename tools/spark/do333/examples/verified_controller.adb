--  STUNIR DO-333 Example: Formally Verified Controller
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Verified_Controller is

   --  ============================================================
   --  Initialize
   --  ============================================================

   procedure Initialize is
   begin
      Current_Altitude := 0;
      Target_Altitude := 0;
      Current_Rate := 0;
      Mode := Mode_Idle;
   end Initialize;

   --  ============================================================
   --  Set Target
   --  ============================================================

   procedure Set_Target (Target : Altitude_Type) is
   begin
      Target_Altitude := Target;

      if Target > Current_Altitude then
         Mode := Mode_Climbing;
      elsif Target < Current_Altitude then
         Mode := Mode_Descending;
      else
         Mode := Mode_Level;
      end if;
   end Set_Target;

   --  ============================================================
   --  Update Altitude
   --  ============================================================

   procedure Update_Altitude (Delta_Time : Positive) is
      Delta_Alt  : Integer;
      New_Alt    : Integer;
      Rate_Limit : constant Rate_Type := Max_Rate;
   begin
      --  Calculate altitude change
      --  Rate is in feet/minute, Delta_Time in seconds
      pragma Assert (Delta_Time <= 60);
      pragma Assert (abs (Current_Rate) <= Max_Rate);

      --  Simplified calculation to avoid overflow
      Delta_Alt := (Current_Rate * Delta_Time) / 60;

      --  Calculate new altitude
      New_Alt := Integer (Current_Altitude) + Delta_Alt;

      --  Clamp to valid range
      if New_Alt < 0 then
         Current_Altitude := 0;
         Current_Rate := 0;
         Mode := Mode_Level;
      elsif New_Alt > Integer (Max_Altitude) then
         Current_Altitude := Max_Altitude;
         Current_Rate := 0;
         Mode := Mode_Level;
      else
         Current_Altitude := Altitude_Type (New_Alt);
      end if;

      --  Update mode based on position relative to target
      if Current_Altitude = Target_Altitude then
         Mode := Mode_Level;
         Current_Rate := 0;
      elsif Mode = Mode_Climbing and then Current_Altitude >= Target_Altitude then
         Mode := Mode_Level;
         Current_Rate := 0;
      elsif Mode = Mode_Descending and then Current_Altitude <= Target_Altitude then
         Mode := Mode_Level;
         Current_Rate := 0;
      end if;

      --  Ensure rate within bounds
      if Current_Rate > Rate_Limit then
         Current_Rate := Rate_Limit;
      elsif Current_Rate < -Rate_Limit then
         Current_Rate := -Rate_Limit;
      end if;
   end Update_Altitude;

   --  ============================================================
   --  Emergency Stop
   --  ============================================================

   procedure Emergency_Stop is
   begin
      Current_Rate := 0;
      Mode := Mode_Level;
   end Emergency_Stop;

   --  ============================================================
   --  Get Altitude
   --  ============================================================

   function Get_Altitude return Altitude_Type is
   begin
      return Current_Altitude;
   end Get_Altitude;

   --  ============================================================
   --  Get Mode
   --  ============================================================

   function Get_Mode return Controller_Mode is
   begin
      return Mode;
   end Get_Mode;

   --  ============================================================
   --  At Target
   --  ============================================================

   function At_Target return Boolean is
   begin
      return Current_Altitude = Target_Altitude;
   end At_Target;

end Verified_Controller;
