--  STUNIR Test Data Generator Types
--  SPARK Migration Phase 3 - Test Infrastructure
--  Test data generation types and contracts

pragma SPARK_Mode (On);

with Stunir_Hashes; use Stunir_Hashes;

package Test_Data_Types is

   --  ===========================================
   --  Constants
   --  ===========================================

   Max_Vectors : constant := 100;
   Max_Vector_Name : constant := 64;
   Max_Input_Size : constant := 4096;
   Max_Output_Size : constant := 4096;

   --  ===========================================
   --  Vector Category
   --  ===========================================

   type Vector_Category is (
      Conformance,    --  Cross-tool conformance
      Boundary,       --  Boundary value testing
      Error_Cat,      --  Error handling testing
      Performance,    --  Performance testing
      Regression,     --  Regression testing
      Random_Cat      --  Randomly generated
   );

   --  ===========================================
   --  Vector Priority
   --  ===========================================

   type Vector_Priority is (Critical, High, Medium, Low);

   --  ===========================================
   --  String Types
   --  ===========================================

   subtype Vector_Name_String is String (1 .. Max_Vector_Name);
   subtype Input_String is String (1 .. Max_Input_Size);
   subtype Output_String is String (1 .. Max_Output_Size);

   --  ===========================================
   --  Test Vector
   --  ===========================================

   type Test_Vector is record
      Name          : Vector_Name_String;
      Name_Len      : Natural;
      Category      : Vector_Category;
      Priority      : Vector_Priority;
      Input_Data    : Input_String;
      Input_Len     : Natural;
      Expected_Hash : String (1 .. Hash_Length);
      Is_Valid      : Boolean;
   end record;

   --  Empty vector constant
   Empty_Vector : constant Test_Vector := (
      Name          => (others => ' '),
      Name_Len      => 0,
      Category      => Conformance,
      Priority      => Medium,
      Input_Data    => (others => ' '),
      Input_Len     => 0,
      Expected_Hash => (others => '0'),
      Is_Valid      => False
   );

   --  ===========================================
   --  Vector Template
   --  ===========================================

   type Template_Kind is (
      Json_Spec,      --  JSON specification template
      IR_Module,      --  IR module template
      Receipt,        --  Receipt template
      Manifest,       --  Manifest template
      Empty           --  Empty/minimal template
   );

   type Vector_Template is record
      Kind       : Template_Kind;
      Category   : Vector_Category;
      Priority   : Vector_Priority;
      Variation  : Natural;  --  Variation number for generation
   end record;

   --  ===========================================
   --  Vector Array
   --  ===========================================

   type Vector_Index is range 0 .. Max_Vectors;
   subtype Valid_Vector_Index is Vector_Index range 1 .. Max_Vectors;

   type Vector_Array is array (Valid_Vector_Index) of Test_Vector;

   --  ===========================================
   --  Vector Set Statistics
   --  ===========================================

   type Vector_Stats is record
      Total       : Natural := 0;
      Conformance : Natural := 0;
      Boundary    : Natural := 0;
      Error_Cnt   : Natural := 0;
      Performance : Natural := 0;
      Regression  : Natural := 0;
      Random_Cnt  : Natural := 0;
   end record;

   --  Empty stats constant
   Empty_Vector_Stats : constant Vector_Stats := (
      Total       => 0,
      Conformance => 0,
      Boundary    => 0,
      Error_Cnt   => 0,
      Performance => 0,
      Regression  => 0,
      Random_Cnt  => 0
   );

   --  ===========================================
   --  Vector Set
   --  ===========================================

   type Vector_Set is record
      Vectors : Vector_Array;
      Count   : Vector_Index;
      Stats   : Vector_Stats;
   end record;

   --  Initialize functions
   function Init_Vector_Array return Vector_Array;
   function Empty_Vector_Set return Vector_Set;

   --  ===========================================
   --  Helper Functions
   --  ===========================================

   function Is_Empty_Vector (V : Test_Vector) return Boolean is
      (V.Name_Len = 0 or not V.Is_Valid);

   function Get_Category_Count (
      Stats : Vector_Stats;
      Cat   : Vector_Category) return Natural;

end Test_Data_Types;
