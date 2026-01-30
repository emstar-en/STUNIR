--  STUNIR DO-331 Interface Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO331_Interface is

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
     (Config     : out Transform_Config;
      IR_Path    : String;
      Output_Path: String;
      DAL        : DAL_Level)
   is
      IR_Len  : Natural;
      Out_Len : Natural;
   begin
      Config := Null_Transform_Config;
      
      Copy_String(IR_Path, Config.IR_Path, IR_Len);
      Config.IR_Path_Len := IR_Path_Length(IR_Len);

      Copy_String(Output_Path, Config.Output_Path, Out_Len);
      Config.Output_Len := Output_Path_Length(Out_Len);

      Config.DAL := DAL;
   end Initialize_Config;

   --  ============================================================
   --  Transform To SysML
   --  ============================================================

   procedure Transform_To_SysML
     (Config : Transform_Config;
      Result : out DO331_Result;
      Status : out DO331_Status)
   is
   begin
      Result := Null_DO331_Result;
      Result.DAL := Config.DAL;

      --  Simulated transformation (real implementation would
      --  call the DO-331 SPARK binary)
      --  Add a sample model entry
      declare
         M : Model_Item := Null_Model_Item;
         Name_Len : Natural;
         Path_Len : Natural;
      begin
         Copy_String("main_module", M.Name, Name_Len);
         M.Name_Len := Model_Name_Length(Name_Len);
         
         Copy_String(Config.Output_Path(1..Config.Output_Len), M.Path, Path_Len);
         M.Path_Len := Model_Path_Length(Path_Len);
         
         M.Kind := Block_Model;
         M.Element_Count := 10;
         M.Is_Valid := True;
         
         Result.Model_Total := 1;
         Result.Models(1) := M;
      end;

      --  Collect coverage if requested
      if Config.Include_Cov then
         Result.Coverage_Pct := 85.0;
      end if;

      Result.Success := True;
      Status := Success;
   end Transform_To_SysML;

   --  ============================================================
   --  Collect Coverage
   --  ============================================================

   procedure Collect_Coverage
     (Model_Path : String;
      Result     : in out DO331_Result;
      Status     : out DO331_Status)
   is
      pragma Unreferenced (Model_Path);
   begin
      --  Simulated coverage collection
      Result.Coverage_Pct := 87.5;
      Status := Success;
   end Collect_Coverage;

   --  ============================================================
   --  Generate Traceability
   --  ============================================================

   procedure Generate_Traceability
     (Config : Transform_Config;
      Result : in out DO331_Result;
      Status : out DO331_Status)
   is
      pragma Unreferenced (Config);
      Link : Trace_Link := Null_Trace_Link;
      Src_Len, Tgt_Len : Natural;
   begin
      --  Add sample traceability link
      Copy_String("REQ-001", Link.Source_ID, Src_Len);
      Link.Source_Len := Element_ID_Length(Src_Len);
      
      Copy_String("IMPL-001", Link.Target_ID, Tgt_Len);
      Link.Target_Len := Element_ID_Length(Tgt_Len);
      
      Link.Direction := Forward;
      Link.Is_Valid := True;

      Result.Trace_Total := 1;
      Result.Trace_Links(1) := Link;
      Status := Success;
   end Generate_Traceability;

   --  ============================================================
   --  Add Model
   --  ============================================================

   procedure Add_Model
     (Result : in out DO331_Result;
      Name   : String;
      Path   : String;
      Kind   : Model_Kind;
      Status : out DO331_Status)
   is
      M : Model_Item := Null_Model_Item;
      Name_Len, Path_Len : Natural;
   begin
      Copy_String(Name, M.Name, Name_Len);
      M.Name_Len := Model_Name_Length(Name_Len);

      Copy_String(Path, M.Path, Path_Len);
      M.Path_Len := Model_Path_Length(Path_Len);

      M.Kind := Kind;
      M.Is_Valid := True;

      Result.Model_Total := Result.Model_Total + 1;
      Result.Models(Result.Model_Total) := M;
      Status := Success;
   end Add_Model;

   --  ============================================================
   --  Add Coverage Item
   --  ============================================================

   procedure Add_Coverage_Item
     (Result     : in out DO331_Result;
      Element_ID : String;
      Kind       : Coverage_Kind;
      Covered    : Boolean;
      Status     : out DO331_Status)
   is
      Item : Coverage_Item := Null_Coverage_Item;
      ID_Len : Natural;
   begin
      Copy_String(Element_ID, Item.Element_ID, ID_Len);
      Item.Element_Len := Element_ID_Length(ID_Len);
      Item.Kind := Kind;
      Item.Covered := Covered;
      Item.Hit_Count := (if Covered then 1 else 0);

      Result.Coverage_Total := Result.Coverage_Total + 1;
      Result.Coverage_Items(Result.Coverage_Total) := Item;
      Status := Success;
   end Add_Coverage_Item;

   --  ============================================================
   --  Add Trace Link
   --  ============================================================

   procedure Add_Trace_Link
     (Result    : in out DO331_Result;
      Source_ID : String;
      Target_ID : String;
      Direction : Trace_Direction;
      Status    : out DO331_Status)
   is
      Link : Trace_Link := Null_Trace_Link;
      Src_Len, Tgt_Len : Natural;
   begin
      Copy_String(Source_ID, Link.Source_ID, Src_Len);
      Link.Source_Len := Element_ID_Length(Src_Len);

      Copy_String(Target_ID, Link.Target_ID, Tgt_Len);
      Link.Target_Len := Element_ID_Length(Tgt_Len);

      Link.Direction := Direction;
      Link.Is_Valid := True;

      Result.Trace_Total := Result.Trace_Total + 1;
      Result.Trace_Links(Result.Trace_Total) := Link;
      Status := Success;
   end Add_Trace_Link;

   --  ============================================================
   --  Validate Model Completeness
   --  ============================================================

   function Validate_Model_Completeness
     (Result : DO331_Result) return Boolean
   is
   begin
      return Result.Model_Total > 0 and
             Result.Success and
             (for all I in 1 .. Result.Model_Total =>
                Result.Models(I).Is_Valid);
   end Validate_Model_Completeness;

   --  ============================================================
   --  Meets DAL Requirements
   --  ============================================================

   function Meets_DAL_Requirements
     (Result : DO331_Result;
      DAL    : DAL_Level) return Boolean
   is
      Min_Coverage : Percentage_Type;
   begin
      --  Minimum coverage requirements by DAL
      case DAL is
         when DAL_A => Min_Coverage := 100.0;
         when DAL_B => Min_Coverage := 95.0;
         when DAL_C => Min_Coverage := 85.0;
         when DAL_D => Min_Coverage := 70.0;
         when DAL_E => Min_Coverage := 0.0;
      end case;

      return Result.Coverage_Pct >= Min_Coverage and
             Result.Success;
   end Meets_DAL_Requirements;

   --  ============================================================
   --  Calculate Coverage Percentage
   --  ============================================================

   function Calculate_Coverage_Percentage
     (Result : DO331_Result) return Percentage_Type
   is
      Covered : Natural := 0;
   begin
      if Result.Coverage_Total = 0 then
         return 0.0;
      end if;

      for I in 1 .. Result.Coverage_Total loop
         pragma Loop_Invariant (Covered <= I - 1);
         if Result.Coverage_Items(I).Covered then
            Covered := Covered + 1;
         end if;
      end loop;

      return Percentage_Type(Float(Covered) / Float(Result.Coverage_Total) * 100.0);
   end Calculate_Coverage_Percentage;

   --  ============================================================
   --  Finalize Result
   --  ============================================================

   procedure Finalize_Result
     (Result : in out DO331_Result)
   is
   begin
      Result.Coverage_Pct := Calculate_Coverage_Percentage(Result);
      Result.Is_Complete := Validate_Model_Completeness(Result);
   end Finalize_Result;

end DO331_Interface;
