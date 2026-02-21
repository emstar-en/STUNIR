--  STUNIR Coverage Analyzer Types
--  SPARK Migration Phase 3 - Test Infrastructure
--  Coverage tracking types and contracts

pragma SPARK_Mode (On);

package Coverage_Types is

   --  ===========================================
   --  Constants
   --  ===========================================

   Max_Modules : constant := 64;
   Max_Lines : constant := 10000;
   Max_Branches : constant := 2000;
   Max_Functions : constant := 500;
   Max_Module_Name : constant := 128;

   --  ===========================================
   --  Percentage Type
   --  ===========================================

   subtype Percentage is Natural range 0 .. 100;

   --  ===========================================
   --  Coverage Thresholds
   --  ===========================================

   Minimum_Line_Coverage     : constant Percentage := 80;
   Minimum_Branch_Coverage   : constant Percentage := 70;
   Minimum_Function_Coverage : constant Percentage := 90;

   --  ===========================================
   --  Coverage Level
   --  ===========================================

   type Coverage_Level is (
      Full,        --  100% coverage
      High,        --  >= 90%
      Medium,      --  >= 70%
      Low,         --  >= 50%
      Minimal,     --  < 50%
      None         --  0%
   );

   --  ===========================================
   --  Module Name Type
   --  ===========================================

   subtype Module_Name_String is String (1 .. Max_Module_Name);

   --  ===========================================
   --  Coverage Metrics
   --  ===========================================

   type Coverage_Metrics is record
      Total_Lines       : Natural := 0;
      Covered_Lines     : Natural := 0;
      Total_Branches    : Natural := 0;
      Covered_Branches  : Natural := 0;
      Total_Functions   : Natural := 0;
      Covered_Functions : Natural := 0;
   end record;

   --  Empty metrics constant
   Empty_Metrics : constant Coverage_Metrics := (
      Total_Lines       => 0,
      Covered_Lines     => 0,
      Total_Branches    => 0,
      Covered_Branches  => 0,
      Total_Functions   => 0,
      Covered_Functions => 0
   );

   --  ===========================================
   --  Line Coverage Bitmap
   --  ===========================================

   type Line_Index is range 0 .. Max_Lines;
   subtype Valid_Line_Index is Line_Index range 1 .. Max_Lines;

   type Line_Coverage_Map is array (Valid_Line_Index) of Boolean
      with Default_Component_Value => False;

   --  ===========================================
   --  Branch Coverage Bitmap
   --  ===========================================

   type Branch_Index is range 0 .. Max_Branches;
   subtype Valid_Branch_Index is Branch_Index range 1 .. Max_Branches;

   type Branch_Coverage_Map is array (Valid_Branch_Index) of Boolean
      with Default_Component_Value => False;

   --  ===========================================
   --  Function Coverage Bitmap
   --  ===========================================

   type Function_Index is range 0 .. Max_Functions;
   subtype Valid_Function_Index is Function_Index range 1 .. Max_Functions;

   type Function_Coverage_Map is array (Valid_Function_Index) of Boolean
      with Default_Component_Value => False;

   --  ===========================================
   --  Module Coverage Data
   --  ===========================================

   type Module_Coverage is record
      Name          : Module_Name_String;
      Name_Len      : Natural;
      Lines         : Line_Coverage_Map;
      Line_Count    : Line_Index;
      Branches      : Branch_Coverage_Map;
      Branch_Count  : Branch_Index;
      Functions     : Function_Coverage_Map;
      Func_Count    : Function_Index;
      Metrics       : Coverage_Metrics;
   end record;

   --  Empty module coverage constant
   Empty_Module_Coverage : constant Module_Coverage := (
      Name         => (others => ' '),
      Name_Len     => 0,
      Lines        => (others => False),
      Line_Count   => 0,
      Branches     => (others => False),
      Branch_Count => 0,
      Functions    => (others => False),
      Func_Count   => 0,
      Metrics      => Empty_Metrics
   );

   --  ===========================================
   --  Module Coverage Array
   --  ===========================================

   type Module_Index is range 0 .. Max_Modules;
   subtype Valid_Module_Index is Module_Index range 1 .. Max_Modules;

   type Module_Array is array (Valid_Module_Index) of Module_Coverage;

   --  ===========================================
   --  Coverage Report
   --  ===========================================

   type Coverage_Report is record
      Modules       : Module_Array;
      Module_Count  : Module_Index;
      Total_Metrics : Coverage_Metrics;
      Line_Pct      : Percentage;
      Branch_Pct    : Percentage;
      Function_Pct  : Percentage;
      Is_Valid      : Boolean;
      Meets_Minimum : Boolean;
   end record;

   --  Initialize function
   function Init_Module_Array return Module_Array;
   function Empty_Report return Coverage_Report;

   --  ===========================================
   --  Helper Functions
   --  ===========================================

   function Get_Line_Coverage (M : Coverage_Metrics) return Percentage
      with Post => Get_Line_Coverage'Result <= 100;

   function Get_Branch_Coverage (M : Coverage_Metrics) return Percentage
      with Post => Get_Branch_Coverage'Result <= 100;

   function Get_Function_Coverage (M : Coverage_Metrics) return Percentage
      with Post => Get_Function_Coverage'Result <= 100;

   function Classify_Coverage (Pct : Percentage) return Coverage_Level;

   function Meets_Minimum_Coverage (M : Coverage_Metrics) return Boolean;

end Coverage_Types;
