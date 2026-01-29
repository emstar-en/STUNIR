--  STUNIR DO-331 Traceability Framework Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Transformer_Utils; use Transformer_Utils;

package body Traceability is

   --  Next trace ID
   Next_Trace_ID : Natural := 1;

   --  ============================================================
   --  Matrix Operations
   --  ============================================================

   function Create_Matrix return Trace_Matrix is
      Result : Trace_Matrix;
   begin
      Result.Entry_Count := 0;
      Result.IR_Hash := (others => '0');
      Result.IR_Hash_Len := 0;
      Result.Model_Hash := (others => '0');
      Result.Model_Hash_Len := 0;
      Result.Created_At := Get_Current_Epoch;
      
      for I in Result.Entries'Range loop
         Result.Entries (I) := Null_Trace_Entry;
      end loop;
      
      return Result;
   end Create_Matrix;

   procedure Add_Trace (
      Matrix    : in Out Trace_Matrix;
      Kind      : in     Trace_Type;
      Source_ID : in     Element_ID;
      Src_Path  : in     String;
      Target_ID : in     Element_ID;
      Tgt_Path  : in     String;
      Rule      : in     Transformation_Rule
   ) is
      Entry : Trace_Entry;
      Src_Len : Natural;
      Tgt_Len : Natural;
   begin
      Entry.Trace_ID := Next_Trace_ID;
      Next_Trace_ID := Next_Trace_ID + 1;
      
      Entry.Trace_Kind := Kind;
      Entry.Source_ID := Source_ID;
      Entry.Target_ID := Target_ID;
      Entry.Rule := Rule;
      Entry.Timestamp := Get_Current_Epoch;
      Entry.Verified := False;
      
      --  Copy source path
      Entry.Source_Path := (others => ' ');
      Src_Len := Natural'Min (Src_Path'Length, Max_Path_Length);
      if Src_Len > 0 then
         Entry.Source_Path (1 .. Src_Len) := Src_Path (Src_Path'First .. Src_Path'First + Src_Len - 1);
      end if;
      Entry.Source_Path_Len := Src_Len;
      
      --  Copy target path
      Entry.Target_Path := (others => ' ');
      Tgt_Len := Natural'Min (Tgt_Path'Length, Max_Path_Length);
      if Tgt_Len > 0 then
         Entry.Target_Path (1 .. Tgt_Len) := Tgt_Path (Tgt_Path'First .. Tgt_Path'First + Tgt_Len - 1);
      end if;
      Entry.Target_Path_Len := Tgt_Len;
      
      Matrix.Entry_Count := Matrix.Entry_Count + 1;
      Matrix.Entries (Matrix.Entry_Count) := Entry;
   end Add_Trace;

   procedure Set_IR_Hash (
      Matrix : in Out Trace_Matrix;
      Hash   : in     String
   ) is
   begin
      Matrix.IR_Hash := (others => '0');
      Matrix.IR_Hash (1 .. Hash'Length) := Hash;
      Matrix.IR_Hash_Len := Hash'Length;
   end Set_IR_Hash;

   procedure Set_Model_Hash (
      Matrix : in Out Trace_Matrix;
      Hash   : in     String
   ) is
   begin
      Matrix.Model_Hash := (others => '0');
      Matrix.Model_Hash (1 .. Hash'Length) := Hash;
      Matrix.Model_Hash_Len := Hash'Length;
   end Set_Model_Hash;

   --  ============================================================
   --  Lookup Operations
   --  ============================================================

   function Get_Forward_Traces (
      Matrix    : Trace_Matrix;
      Source_ID : Element_ID
   ) return Trace_Entry_Array is
      Count : Natural := 0;
   begin
      --  Count matching entries
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Source_ID = Source_ID then
            Count := Count + 1;
         end if;
      end loop;
      
      --  Build result array
      declare
         Result : Trace_Entry_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Matrix.Entry_Count loop
            if Matrix.Entries (I).Source_ID = Source_ID then
               J := J + 1;
               Result (J) := Matrix.Entries (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Forward_Traces;

   function Get_Backward_Traces (
      Matrix    : Trace_Matrix;
      Target_ID : Element_ID
   ) return Trace_Entry_Array is
      Count : Natural := 0;
   begin
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Target_ID = Target_ID then
            Count := Count + 1;
         end if;
      end loop;
      
      declare
         Result : Trace_Entry_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Matrix.Entry_Count loop
            if Matrix.Entries (I).Target_ID = Target_ID then
               J := J + 1;
               Result (J) := Matrix.Entries (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Backward_Traces;

   function Get_Traces_By_Rule (
      Matrix : Trace_Matrix;
      Rule   : Transformation_Rule
   ) return Trace_Entry_Array is
      Count : Natural := 0;
   begin
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Rule = Rule then
            Count := Count + 1;
         end if;
      end loop;
      
      declare
         Result : Trace_Entry_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Matrix.Entry_Count loop
            if Matrix.Entries (I).Rule = Rule then
               J := J + 1;
               Result (J) := Matrix.Entries (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Traces_By_Rule;

   function Has_Trace (
      Matrix : Trace_Matrix;
      ID     : Element_ID
   ) return Boolean is
   begin
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Source_ID = ID or
            Matrix.Entries (I).Target_ID = ID
         then
            return True;
         end if;
      end loop;
      return False;
   end Has_Trace;

   --  ============================================================
   --  Completeness Analysis
   --  ============================================================

   function Check_Completeness (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Boolean is
   begin
      for I in IR_IDs'Range loop
         if not Has_Trace (Matrix, IR_IDs (I)) then
            return False;
         end if;
      end loop;
      return True;
   end Check_Completeness;

   function Analyze_Gaps (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Gap_Report is
      Report  : Gap_Report;
      Traced  : Natural := 0;
      Missing : Natural := 0;
   begin
      Report.Total_IR_Elements := IR_IDs'Length;
      
      for I in IR_IDs'Range loop
         if Has_Trace (Matrix, IR_IDs (I)) then
            Traced := Traced + 1;
         else
            Missing := Missing + 1;
         end if;
      end loop;
      
      Report.Traced_Elements := Traced;
      Report.Missing_Traces := Missing;
      
      if IR_IDs'Length > 0 then
         Report.Gap_Percentage := (Missing * 100) / IR_IDs'Length;
      else
         Report.Gap_Percentage := 0;
      end if;
      
      Report.Is_Complete := Missing = 0;
      
      return Report;
   end Analyze_Gaps;

   function Get_Missing_Traces (
      Matrix : Trace_Matrix;
      IR_IDs : Element_ID_Array
   ) return Element_ID_Array is
      Count : Natural := 0;
   begin
      --  Count missing
      for I in IR_IDs'Range loop
         if not Has_Trace (Matrix, IR_IDs (I)) then
            Count := Count + 1;
         end if;
      end loop;
      
      --  Build result
      declare
         Result : Element_ID_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in IR_IDs'Range loop
            if not Has_Trace (Matrix, IR_IDs (I)) then
               J := J + 1;
               Result (J) := IR_IDs (I);
            end if;
         end loop;
         return Result;
      end;
   end Get_Missing_Traces;

   --  ============================================================
   --  Verification
   --  ============================================================

   procedure Mark_Verified (
      Matrix   : in Out Trace_Matrix;
      Trace_ID : in     Natural
   ) is
   begin
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Trace_ID = Trace_ID then
            Matrix.Entries (I).Verified := True;
            exit;
         end if;
      end loop;
   end Mark_Verified;

   function Get_Verification_Status (
      Matrix : Trace_Matrix
   ) return Natural is
      Verified : Natural := 0;
   begin
      if Matrix.Entry_Count = 0 then
         return 100;
      end if;
      
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Verified then
            Verified := Verified + 1;
         end if;
      end loop;
      
      return (Verified * 100) / Matrix.Entry_Count;
   end Get_Verification_Status;

   --  ============================================================
   --  Validation
   --  ============================================================

   function Validate_Matrix (Matrix : Trace_Matrix) return Boolean is
   begin
      --  Check basic integrity
      if Matrix.Entry_Count > Max_Trace_Entries then
         return False;
      end if;
      
      --  Check all entries have valid IDs
      for I in 1 .. Matrix.Entry_Count loop
         if Matrix.Entries (I).Source_ID = Null_Element_ID and
            Matrix.Entries (I).Target_ID = Null_Element_ID
         then
            return False;
         end if;
      end loop;
      
      return True;
   end Validate_Matrix;

end Traceability;
