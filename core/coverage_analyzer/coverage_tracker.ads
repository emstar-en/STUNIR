--  STUNIR Coverage Tracker
--  SPARK Migration Phase 3 - Test Infrastructure
--  Tracks code coverage for test validation

pragma SPARK_Mode (On);

with Coverage_Types; use Coverage_Types;

package Coverage_Tracker is

   --  ===========================================
   --  Tracker Type
   --  ===========================================

   type Coverage_Tracker_Type is record
      Modules      : Module_Array;
      Module_Count : Module_Index;
      Is_Active    : Boolean;
   end record;

   --  Empty tracker function
   function Empty_Tracker return Coverage_Tracker_Type;

   --  ===========================================
   --  Initialization
   --  ===========================================

   --  Initialize coverage tracker
   procedure Initialize (Tracker : out Coverage_Tracker_Type)
      with Post => Tracker.Module_Count = 0 and not Tracker.Is_Active;

   --  Start tracking
   procedure Start_Tracking (Tracker : in out Coverage_Tracker_Type)
      with Post => Tracker.Is_Active;

   --  Stop tracking
   procedure Stop_Tracking (Tracker : in out Coverage_Tracker_Type)
      with Post => not Tracker.Is_Active;

   --  ===========================================
   --  Module Management
   --  ===========================================

   --  Register a module for coverage tracking
   procedure Register_Module (
      Tracker    : in out Coverage_Tracker_Type;
      Name       : in String;
      Line_Count : in Natural;
      Success    : out Boolean)
      with Pre  => Tracker.Module_Count < Max_Modules and
                   Name'Length > 0 and
                   Name'Length <= Max_Module_Name and
                   Line_Count <= Natural (Max_Lines),
           Post => (if Success then
                       Tracker.Module_Count = Tracker.Module_Count'Old + 1
                    else Tracker.Module_Count = Tracker.Module_Count'Old);

   --  Find module by name
   function Find_Module (
      Tracker : Coverage_Tracker_Type;
      Name    : String) return Module_Index
      with Pre => Name'Length > 0 and Name'Length <= Max_Module_Name;

   --  ===========================================
   --  Line Coverage Recording
   --  ===========================================

   --  Record line execution
   procedure Record_Line (
      Tracker  : in out Coverage_Tracker_Type;
      Module   : in Valid_Module_Index;
      Line     : in Positive;
      Executed : in Boolean)
      with Pre => Module <= Tracker.Module_Count and
                  Line <= Natural (Max_Lines);

   --  Mark line range as covered
   procedure Mark_Lines_Covered (
      Tracker    : in out Coverage_Tracker_Type;
      Module     : in Valid_Module_Index;
      Start_Line : in Positive;
      End_Line   : in Positive)
      with Pre => Module <= Tracker.Module_Count and
                  Start_Line <= End_Line and
                  End_Line <= Natural (Max_Lines);

   --  ===========================================
   --  Branch Coverage Recording
   --  ===========================================

   --  Record branch taken
   procedure Record_Branch (
      Tracker : in out Coverage_Tracker_Type;
      Module  : in Valid_Module_Index;
      Branch  : in Positive;
      Taken   : in Boolean)
      with Pre => Module <= Tracker.Module_Count and
                  Branch <= Natural (Max_Branches);

   --  ===========================================
   --  Function Coverage Recording
   --  ===========================================

   --  Record function call
   procedure Record_Function (
      Tracker  : in out Coverage_Tracker_Type;
      Module   : in Valid_Module_Index;
      Func     : in Positive;
      Called   : in Boolean)
      with Pre => Module <= Tracker.Module_Count and
                  Func <= Natural (Max_Functions);

   --  ===========================================
   --  Metrics Computation
   --  ===========================================

   --  Compute metrics for a single module
   procedure Compute_Module_Metrics (
      Tracker : in out Coverage_Tracker_Type;
      Module  : in Valid_Module_Index)
      with Pre => Module <= Tracker.Module_Count;

   --  Compute all module metrics
   procedure Compute_All_Metrics (
      Tracker : in out Coverage_Tracker_Type);

   --  ===========================================
   --  Report Generation
   --  ===========================================

   --  Generate coverage report
   function Get_Report (
      Tracker : Coverage_Tracker_Type) return Coverage_Report
      with Post => Get_Report'Result.Is_Valid;

   --  Get module coverage
   function Get_Module_Coverage (
      Tracker : Coverage_Tracker_Type;
      Module  : Valid_Module_Index) return Coverage_Metrics
      with Pre => Module <= Tracker.Module_Count;

   --  Check if coverage meets minimum requirements
   function Meets_Requirements (
      Tracker : Coverage_Tracker_Type) return Boolean;

end Coverage_Tracker;
