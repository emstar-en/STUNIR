--  STUNIR Test Data Generator
--  SPARK Migration Phase 3 - Test Infrastructure
--  Generates test vectors for conformance testing

pragma SPARK_Mode (On);

with Test_Data_Types; use Test_Data_Types;
with Stunir_Hashes; use Stunir_Hashes;

package Data_Generator is

   --  ===========================================
   --  Vector Set Management
   --  ===========================================

   --  Initialize empty vector set
   procedure Initialize_Set (VSet : out Vector_Set)
      with Post => VSet.Count = 0 and VSet.Stats = Empty_Vector_Stats;

   --  Add vector to set
   procedure Add_Vector (
      VSet    : in out Vector_Set;
      Vector  : in Test_Vector;
      Success : out Boolean)
      with Pre  => VSet.Count < Max_Vectors and Vector.Is_Valid,
           Post => (if Success then
                       VSet.Count = VSet.Count'Old + 1
                    else VSet.Count = VSet.Count'Old);

   --  Get vector by index
   function Get_Vector (
      VSet  : Vector_Set;
      Index : Valid_Vector_Index) return Test_Vector
      with Pre => Index <= VSet.Count;

   --  ===========================================
   --  Vector Generation
   --  ===========================================

   --  Create a test vector
   function Create_Vector (
      Name     : String;
      Input    : String;
      Category : Vector_Category;
      Priority : Vector_Priority) return Test_Vector
      with Pre  => Name'Length > 0 and Name'Length <= Max_Vector_Name and
                   Input'Length > 0 and Input'Length <= Max_Input_Size,
           Post => Create_Vector'Result.Is_Valid and
                   Create_Vector'Result.Name_Len = Name'Length and
                   Create_Vector'Result.Input_Len = Input'Length;

   --  Generate vector from template
   procedure Generate_From_Template (
      Template : in Vector_Template;
      Output   : out Test_Vector;
      Success  : out Boolean)
      with Post => (if Success then Output.Is_Valid
                    else not Output.Is_Valid);

   --  ===========================================
   --  Template-Based Generation
   --  ===========================================

   --  Generate JSON spec vector
   procedure Generate_Json_Spec_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
      with Pre  => Name'Length > 0 and Name'Length <= Max_Vector_Name,
           Post => (if Success then Output.Is_Valid);

   --  Generate IR module vector
   procedure Generate_IR_Module_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
      with Pre  => Name'Length > 0 and Name'Length <= Max_Vector_Name,
           Post => (if Success then Output.Is_Valid);

   --  Generate receipt vector
   procedure Generate_Receipt_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
      with Pre  => Name'Length > 0 and Name'Length <= Max_Vector_Name,
           Post => (if Success then Output.Is_Valid);

   --  ===========================================
   --  Boundary Value Generation
   --  ===========================================

   --  Generate boundary test vectors
   procedure Generate_Boundary_Vectors (
      VSet    : in out Vector_Set;
      Added   : out Natural)
      with Pre  => VSet.Count + 10 < Max_Vectors,
           Post => VSet.Count >= VSet.Count'Old and Added <= 10;

   --  Generate minimal vector (empty/edge case)
   function Generate_Minimal_Vector (
      Name : String) return Test_Vector
      with Pre => Name'Length > 0 and Name'Length <= Max_Vector_Name;

   --  ===========================================
   --  Statistics Update
   --  ===========================================

   --  Update statistics for added vector
   procedure Update_Stats (
      Stats    : in out Vector_Stats;
      Category : in Vector_Category)
      with Pre => Stats.Total < Natural'Last,
           Post => Stats.Total = Stats.Total'Old + 1;

   --  ===========================================
   --  Hash Computation
   --  ===========================================

   --  Compute expected hash for input data
   function Compute_Expected_Hash (Input : String) return String
      with Pre  => Input'Length > 0,
           Post => Compute_Expected_Hash'Result'Length = Hash_Length;

end Data_Generator;
