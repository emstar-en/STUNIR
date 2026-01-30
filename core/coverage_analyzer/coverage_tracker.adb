--  STUNIR Coverage Tracker - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Coverage_Tracker is

   --  ===========================================
   --  Empty Tracker
   --  ===========================================

   function Empty_Tracker return Coverage_Tracker_Type is
   begin
      return (
         Modules      => Init_Module_Array,
         Module_Count => 0,
         Is_Active    => False
      );
   end Empty_Tracker;

   --  ===========================================
   --  Initialize
   --  ===========================================

   procedure Initialize (Tracker : out Coverage_Tracker_Type) is
   begin
      Tracker := Empty_Tracker;
   end Initialize;

   --  ===========================================
   --  Start Tracking
   --  ===========================================

   procedure Start_Tracking (Tracker : in out Coverage_Tracker_Type) is
   begin
      Tracker.Is_Active := True;
   end Start_Tracking;

   --  ===========================================
   --  Stop Tracking
   --  ===========================================

   procedure Stop_Tracking (Tracker : in out Coverage_Tracker_Type) is
   begin
      Tracker.Is_Active := False;
   end Stop_Tracking;

   --  ===========================================
   --  Register Module
   --  ===========================================

   procedure Register_Module (
      Tracker    : in out Coverage_Tracker_Type;
      Name       : in String;
      Line_Count : in Natural;
      Success    : out Boolean)
   is
      MC : Module_Coverage := Empty_Module_Coverage;
   begin
      if Tracker.Module_Count = Max_Modules then
         Success := False;
         return;
      end if;

      --  Copy name
      for I in Name'Range loop
         MC.Name (I - Name'First + 1) := Name (I);
      end loop;
      MC.Name_Len := Name'Length;
      MC.Line_Count := Line_Index (Line_Count);

      Tracker.Module_Count := Tracker.Module_Count + 1;
      Tracker.Modules (Valid_Module_Index (Tracker.Module_Count)) := MC;
      Success := True;
   end Register_Module;

   --  ===========================================
   --  Find Module
   --  ===========================================

   function Find_Module (
      Tracker : Coverage_Tracker_Type;
      Name    : String) return Module_Index
   is
      Found : Module_Index := 0;
   begin
      for I in 1 .. Tracker.Module_Count loop
         declare
            M : Module_Coverage renames
               Tracker.Modules (Valid_Module_Index (I));
         begin
            if M.Name_Len = Name'Length then
               declare
                  Match : Boolean := True;
               begin
                  for J in 1 .. M.Name_Len loop
                     if M.Name (J) /= Name (Name'First + J - 1) then
                        Match := False;
                        exit;
                     end if;
                  end loop;
                  if Match then
                     Found := I;
                     exit;
                  end if;
               end;
            end if;
         end;
      end loop;
      return Found;
   end Find_Module;

   --  ===========================================
   --  Record Line
   --  ===========================================

   procedure Record_Line (
      Tracker  : in out Coverage_Tracker_Type;
      Module   : in Valid_Module_Index;
      Line     : in Positive;
      Executed : in Boolean) is
   begin
      if Line <= Natural (Max_Lines) then
         Tracker.Modules (Module).Lines (Valid_Line_Index (Line)) := Executed;
      end if;
   end Record_Line;

   --  ===========================================
   --  Mark Lines Covered
   --  ===========================================

   procedure Mark_Lines_Covered (
      Tracker    : in out Coverage_Tracker_Type;
      Module     : in Valid_Module_Index;
      Start_Line : in Positive;
      End_Line   : in Positive) is
   begin
      for L in Start_Line .. End_Line loop
         if L <= Natural (Max_Lines) then
            Tracker.Modules (Module).Lines (Valid_Line_Index (L)) := True;
         end if;
      end loop;
   end Mark_Lines_Covered;

   --  ===========================================
   --  Record Branch
   --  ===========================================

   procedure Record_Branch (
      Tracker : in out Coverage_Tracker_Type;
      Module  : in Valid_Module_Index;
      Branch  : in Positive;
      Taken   : in Boolean) is
   begin
      if Branch <= Natural (Max_Branches) then
         Tracker.Modules (Module).Branches (Valid_Branch_Index (Branch)) :=
            Taken;
      end if;
   end Record_Branch;

   --  ===========================================
   --  Record Function
   --  ===========================================

   procedure Record_Function (
      Tracker  : in out Coverage_Tracker_Type;
      Module   : in Valid_Module_Index;
      Func     : in Positive;
      Called   : in Boolean) is
   begin
      if Func <= Natural (Max_Functions) then
         Tracker.Modules (Module).Functions (Valid_Function_Index (Func)) :=
            Called;
      end if;
   end Record_Function;

   --  ===========================================
   --  Compute Module Metrics
   --  ===========================================

   procedure Compute_Module_Metrics (
      Tracker : in out Coverage_Tracker_Type;
      Module  : in Valid_Module_Index)
   is
      M : Module_Coverage renames Tracker.Modules (Module);
      Cov_Lines : Natural := 0;
      Cov_Branches : Natural := 0;
      Cov_Functions : Natural := 0;
   begin
      --  Count covered lines
      for I in 1 .. M.Line_Count loop
         if M.Lines (Valid_Line_Index (I)) then
            Cov_Lines := Cov_Lines + 1;
         end if;
      end loop;

      --  Count covered branches
      for I in 1 .. M.Branch_Count loop
         if M.Branches (Valid_Branch_Index (I)) then
            Cov_Branches := Cov_Branches + 1;
         end if;
      end loop;

      --  Count covered functions
      for I in 1 .. M.Func_Count loop
         if M.Functions (Valid_Function_Index (I)) then
            Cov_Functions := Cov_Functions + 1;
         end if;
      end loop;

      --  Update metrics
      M.Metrics.Total_Lines := Natural (M.Line_Count);
      M.Metrics.Covered_Lines := Cov_Lines;
      M.Metrics.Total_Branches := Natural (M.Branch_Count);
      M.Metrics.Covered_Branches := Cov_Branches;
      M.Metrics.Total_Functions := Natural (M.Func_Count);
      M.Metrics.Covered_Functions := Cov_Functions;

      Tracker.Modules (Module) := M;
   end Compute_Module_Metrics;

   --  ===========================================
   --  Compute All Metrics
   --  ===========================================

   procedure Compute_All_Metrics (
      Tracker : in out Coverage_Tracker_Type) is
   begin
      for I in 1 .. Tracker.Module_Count loop
         Compute_Module_Metrics (Tracker, Valid_Module_Index (I));
      end loop;
   end Compute_All_Metrics;

   --  ===========================================
   --  Get Report
   --  ===========================================

   function Get_Report (
      Tracker : Coverage_Tracker_Type) return Coverage_Report
   is
      Report : Coverage_Report := Empty_Report;
      Tot_Lines : Natural := 0;
      Cov_Lines : Natural := 0;
      Tot_Branches : Natural := 0;
      Cov_Branches : Natural := 0;
      Tot_Functions : Natural := 0;
      Cov_Functions : Natural := 0;
   begin
      Report.Modules := Tracker.Modules;
      Report.Module_Count := Tracker.Module_Count;

      --  Aggregate metrics
      for I in 1 .. Tracker.Module_Count loop
         declare
            M : Coverage_Metrics renames
               Tracker.Modules (Valid_Module_Index (I)).Metrics;
         begin
            if Tot_Lines < Natural'Last - M.Total_Lines then
               Tot_Lines := Tot_Lines + M.Total_Lines;
            end if;
            if Cov_Lines < Natural'Last - M.Covered_Lines then
               Cov_Lines := Cov_Lines + M.Covered_Lines;
            end if;
            if Tot_Branches < Natural'Last - M.Total_Branches then
               Tot_Branches := Tot_Branches + M.Total_Branches;
            end if;
            if Cov_Branches < Natural'Last - M.Covered_Branches then
               Cov_Branches := Cov_Branches + M.Covered_Branches;
            end if;
            if Tot_Functions < Natural'Last - M.Total_Functions then
               Tot_Functions := Tot_Functions + M.Total_Functions;
            end if;
            if Cov_Functions < Natural'Last - M.Covered_Functions then
               Cov_Functions := Cov_Functions + M.Covered_Functions;
            end if;
         end;
      end loop;

      Report.Total_Metrics := (
         Total_Lines       => Tot_Lines,
         Covered_Lines     => Cov_Lines,
         Total_Branches    => Tot_Branches,
         Covered_Branches  => Cov_Branches,
         Total_Functions   => Tot_Functions,
         Covered_Functions => Cov_Functions
      );

      Report.Line_Pct := Get_Line_Coverage (Report.Total_Metrics);
      Report.Branch_Pct := Get_Branch_Coverage (Report.Total_Metrics);
      Report.Function_Pct := Get_Function_Coverage (Report.Total_Metrics);
      Report.Is_Valid := True;
      Report.Meets_Minimum := Meets_Minimum_Coverage (Report.Total_Metrics);

      return Report;
   end Get_Report;

   --  ===========================================
   --  Get Module Coverage
   --  ===========================================

   function Get_Module_Coverage (
      Tracker : Coverage_Tracker_Type;
      Module  : Valid_Module_Index) return Coverage_Metrics is
   begin
      return Tracker.Modules (Module).Metrics;
   end Get_Module_Coverage;

   --  ===========================================
   --  Meets Requirements
   --  ===========================================

   function Meets_Requirements (
      Tracker : Coverage_Tracker_Type) return Boolean
   is
      Report : constant Coverage_Report := Get_Report (Tracker);
   begin
      return Report.Meets_Minimum;
   end Meets_Requirements;

end Coverage_Tracker;
