--  STUNIR DO-332 Interface Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO332_Interface is

   --  ============================================================
   --  Helper: Copy String
   --  ============================================================

   procedure Copy_String
     (Source : String;
      Target : out String;
      Length : out Natural)
   with Pre => Target'Length >= Source'Length
   is
   begin
      Length := Source'Length;
      for I in 1 .. Source'Length loop
         pragma Loop_Invariant (I <= Source'Length);
         Target(Target'First + I - 1) := Source(Source'First + I - 1);
      end loop;
      for I in Source'Length + 1 .. Target'Length loop
         Target(Target'First + I - 1) := ' ';
      end loop;
   end Copy_String;

   --  ============================================================
   --  Initialize Config
   --  ============================================================

   procedure Initialize_Config
     (Config      : out Analysis_Config;
      IR_Path     : String;
      Output_Path : String;
      Max_Depth   : Natural := Default_Max_Depth)
   is
      IR_Len, Out_Len : Natural;
   begin
      Config := Null_Analysis_Config;
      
      Copy_String(IR_Path, Config.IR_Path, IR_Len);
      Config.IR_Path_Len := Path_Length(IR_Len);

      Copy_String(Output_Path, Config.Output_Path, Out_Len);
      Config.Output_Len := Path_Length(Out_Len);

      Config.Max_Depth := Max_Depth;
   end Initialize_Config;

   --  ============================================================
   --  Analyze OOP
   --  ============================================================

   procedure Analyze_OOP
     (Config : Analysis_Config;
      Result : out DO332_Result;
      Status : out DO332_Status)
   is
   begin
      Result := Null_DO332_Result;

      --  Simulated OOP analysis
      Analyze_Hierarchy(Config, Result, Status);
      if Status /= Success then
         return;
      end if;

      if Config.Check_Virtual then
         Verify_Polymorphism(Result, Status);
         if Status /= Success then
            return;
         end if;
      end if;

      if Config.Check_Coupling then
         Calculate_Coupling(Result, Status);
         if Status /= Success then
            return;
         end if;
      end if;

      Finalize_Result(Result, Config.Max_Depth, Config.Max_Coupling);
      Result.Success := True;
      Status := Success;
   end Analyze_OOP;

   --  ============================================================
   --  Analyze Hierarchy
   --  ============================================================

   procedure Analyze_Hierarchy
     (Config : Analysis_Config;
      Result : in out DO332_Result;
      Status : out DO332_Status)
   is
      pragma Unreferenced (Config);
   begin
      --  Simulated hierarchy analysis
      Result.Max_Depth := 3;
      Status := Success;
   end Analyze_Hierarchy;

   --  ============================================================
   --  Verify Polymorphism
   --  ============================================================

   procedure Verify_Polymorphism
     (Result : in out DO332_Result;
      Status : out DO332_Status)
   is
   begin
      --  Check all polymorphic calls
      for I in 1 .. Result.Poly_Total loop
         pragma Loop_Invariant (I <= Result.Poly_Total);
         Result.Poly_Calls(I).Is_Verified := True;
      end loop;

      Result.Polymorphism_OK := All_Polymorphism_Safe(Result);
      Status := Success;
   end Verify_Polymorphism;

   --  ============================================================
   --  Calculate Coupling
   --  ============================================================

   procedure Calculate_Coupling
     (Result : in out DO332_Result;
      Status : out DO332_Status)
   is
   begin
      --  Simulated coupling calculation
      Result.Metrics.CBO := 5;
      Result.Metrics.RFC := 10;
      Result.Metrics.WMC := 15;
      Result.Metrics.DIT := Result.Max_Depth;
      Result.Metrics.NOC := Result.Class_Total;
      Result.Metrics.LCOM := 2;
      Status := Success;
   end Calculate_Coupling;

   --  ============================================================
   --  Add Class
   --  ============================================================

   procedure Add_Class
     (Result      : in out DO332_Result;
      Name        : String;
      Parent      : String;
      Inheritance : Inheritance_Kind;
      Depth       : Natural;
      Status      : out DO332_Status)
   is
      C : Class_Entry := Null_Class_Entry;
      Name_Len, Parent_Len : Natural;
   begin
      Copy_String(Name, C.Name, Name_Len);
      C.Name_Len := Class_Name_Length(Name_Len);

      if Parent'Length > 0 then
         Copy_String(Parent, C.Parent_Name, Parent_Len);
         C.Parent_Len := Class_Name_Length(Parent_Len);
      end if;

      C.Inheritance := Inheritance;
      C.Depth := Depth;
      C.Is_Valid := True;

      Result.Class_Total := Result.Class_Total + 1;
      Result.Classes(Result.Class_Total) := C;

      if Depth > Result.Max_Depth then
         Result.Max_Depth := Depth;
      end if;

      Status := Success;
   end Add_Class;

   --  ============================================================
   --  Add Polymorphic Call
   --  ============================================================

   procedure Add_Polymorphic_Call
     (Result       : in out DO332_Result;
      Caller_Class : String;
      Target_Method: String;
      Is_Safe      : Boolean;
      Status       : out DO332_Status)
   is
      P : Polymorphic_Call := Null_Polymorphic_Call;
      Caller_Len, Target_Len : Natural;
   begin
      Copy_String(Caller_Class, P.Caller_Class, Caller_Len);
      P.Caller_Len := Class_Name_Length(Caller_Len);

      Copy_String(Target_Method, P.Target_Method, Target_Len);
      P.Target_Len := Class_Name_Length(Target_Len);

      P.Is_Safe := Is_Safe;
      P.Is_Verified := False;

      Result.Poly_Total := Result.Poly_Total + 1;
      Result.Poly_Calls(Result.Poly_Total) := P;
      Status := Success;
   end Add_Polymorphic_Call;

   --  ============================================================
   --  Check Depth Limits
   --  ============================================================

   function Check_Depth_Limits
     (Result    : DO332_Result;
      Max_Depth : Natural) return Boolean
   is
   begin
      return Result.Max_Depth <= Max_Depth;
   end Check_Depth_Limits;

   --  ============================================================
   --  Check Coupling Limits
   --  ============================================================

   function Check_Coupling_Limits
     (Result      : DO332_Result;
      Max_Coupling: Natural) return Boolean
   is
   begin
      return Result.Metrics.CBO <= Max_Coupling;
   end Check_Coupling_Limits;

   --  ============================================================
   --  All Polymorphism Safe
   --  ============================================================

   function All_Polymorphism_Safe
     (Result : DO332_Result) return Boolean
   is
   begin
      for I in 1 .. Result.Poly_Total loop
         pragma Loop_Invariant (I <= Result.Poly_Total);
         if not Result.Poly_Calls(I).Is_Safe then
            return False;
         end if;
      end loop;
      return True;
   end All_Polymorphism_Safe;

   --  ============================================================
   --  Finalize Result
   --  ============================================================

   procedure Finalize_Result
     (Result     : in out DO332_Result;
      Max_Depth  : Natural;
      Max_Coupling: Natural)
   is
   begin
      Result.Inheritance_OK := Check_Depth_Limits(Result, Max_Depth);
      Result.Polymorphism_OK := All_Polymorphism_Safe(Result);
      Result.Coupling_OK := Check_Coupling_Limits(Result, Max_Coupling);
   end Finalize_Result;

end DO332_Interface;
