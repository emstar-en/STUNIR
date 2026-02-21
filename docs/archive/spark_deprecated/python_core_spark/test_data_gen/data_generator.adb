--  STUNIR Test Data Generator - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

package body Data_Generator is

   --  Template strings for vector generation
   Json_Template : constant String :=
      "{""schema"":""stunir.spec.v1"",""id"":""test""}";

   IR_Template : constant String :=
      "{""schema"":""stunir.ir.v1"",""module"":""test""}";

   Receipt_Template : constant String :=
      "{""schema"":""stunir.receipt.v1"",""outputs"":{}}";

   --  ===========================================
   --  Initialize Set
   --  ===========================================

   procedure Initialize_Set (VSet : out Vector_Set) is
   begin
      VSet := Empty_Vector_Set;
   end Initialize_Set;

   --  ===========================================
   --  Add Vector
   --  ===========================================

   procedure Add_Vector (
      VSet    : in out Vector_Set;
      Vector  : in Test_Vector;
      Success : out Boolean) is
   begin
      if VSet.Count = Max_Vectors or not Vector.Is_Valid then
         Success := False;
         return;
      end if;

      VSet.Count := VSet.Count + 1;
      VSet.Vectors (Valid_Vector_Index (VSet.Count)) := Vector;
      Update_Stats (VSet.Stats, Vector.Category);
      Success := True;
   end Add_Vector;

   --  ===========================================
   --  Get Vector
   --  ===========================================

   function Get_Vector (
      VSet  : Vector_Set;
      Index : Valid_Vector_Index) return Test_Vector is
   begin
      return VSet.Vectors (Index);
   end Get_Vector;

   --  ===========================================
   --  Create Vector
   --  ===========================================

   function Create_Vector (
      Name     : String;
      Input    : String;
      Category : Vector_Category;
      Priority : Vector_Priority) return Test_Vector
   is
      V : Test_Vector := Empty_Vector;
   begin
      --  Copy name
      for I in Name'Range loop
         V.Name (I - Name'First + 1) := Name (I);
      end loop;
      V.Name_Len := Name'Length;

      --  Copy input
      for I in Input'Range loop
         V.Input_Data (I - Input'First + 1) := Input (I);
      end loop;
      V.Input_Len := Input'Length;

      V.Category := Category;
      V.Priority := Priority;
      V.Expected_Hash := Compute_Expected_Hash (Input);
      V.Is_Valid := True;

      return V;
   end Create_Vector;

   --  ===========================================
   --  Generate From Template
   --  ===========================================

   procedure Generate_From_Template (
      Template : in Vector_Template;
      Output   : out Test_Vector;
      Success  : out Boolean)
   is
      Name : constant String := "template_vector";
   begin
      case Template.Kind is
         when Json_Spec =>
            Generate_Json_Spec_Vector (Name, Template.Variation,
                                       Output, Success);
         when IR_Module =>
            Generate_IR_Module_Vector (Name, Template.Variation,
                                       Output, Success);
         when Receipt =>
            Generate_Receipt_Vector (Name, Template.Variation,
                                     Output, Success);
         when Manifest =>
            Generate_Receipt_Vector (Name, Template.Variation,
                                     Output, Success);
         when Empty =>
            Output := Generate_Minimal_Vector (Name);
            Success := Output.Is_Valid;
      end case;

      if Success then
         Output.Category := Template.Category;
         Output.Priority := Template.Priority;
      end if;
   end Generate_From_Template;

   --  ===========================================
   --  Generate JSON Spec Vector
   --  ===========================================

   procedure Generate_Json_Spec_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
   is
      pragma Unreferenced (Variant);
   begin
      if Json_Template'Length > Max_Input_Size then
         Output := Empty_Vector;
         Success := False;
         return;
      end if;

      Output := Create_Vector (
         Name     => Name,
         Input    => Json_Template,
         Category => Conformance,
         Priority => High
      );
      Success := True;
   end Generate_Json_Spec_Vector;

   --  ===========================================
   --  Generate IR Module Vector
   --  ===========================================

   procedure Generate_IR_Module_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
   is
      pragma Unreferenced (Variant);
   begin
      if IR_Template'Length > Max_Input_Size then
         Output := Empty_Vector;
         Success := False;
         return;
      end if;

      Output := Create_Vector (
         Name     => Name,
         Input    => IR_Template,
         Category => Conformance,
         Priority => High
      );
      Success := True;
   end Generate_IR_Module_Vector;

   --  ===========================================
   --  Generate Receipt Vector
   --  ===========================================

   procedure Generate_Receipt_Vector (
      Name     : in String;
      Variant  : in Natural;
      Output   : out Test_Vector;
      Success  : out Boolean)
   is
      pragma Unreferenced (Variant);
   begin
      if Receipt_Template'Length > Max_Input_Size then
         Output := Empty_Vector;
         Success := False;
         return;
      end if;

      Output := Create_Vector (
         Name     => Name,
         Input    => Receipt_Template,
         Category => Conformance,
         Priority => High
      );
      Success := True;
   end Generate_Receipt_Vector;

   --  ===========================================
   --  Generate Boundary Vectors
   --  ===========================================

   procedure Generate_Boundary_Vectors (
      VSet    : in out Vector_Set;
      Added   : out Natural)
   is
      V : Test_Vector;
      Success : Boolean;
      Names : constant array (1 .. 5) of String (1 .. 10) := (
         "boundary_1",
         "boundary_2",
         "boundary_3",
         "boundary_4",
         "boundary_5"
      );
   begin
      Added := 0;

      for I in Names'Range loop
         exit when VSet.Count = Max_Vectors;

         V := Create_Vector (
            Name     => Names (I),
            Input    => "{}",
            Category => Boundary,
            Priority => High
         );

         Add_Vector (VSet, V, Success);
         if Success then
            Added := Added + 1;
         end if;
      end loop;
   end Generate_Boundary_Vectors;

   --  ===========================================
   --  Generate Minimal Vector
   --  ===========================================

   function Generate_Minimal_Vector (
      Name : String) return Test_Vector is
   begin
      return Create_Vector (
         Name     => Name,
         Input    => "{}",
         Category => Boundary,
         Priority => Low
      );
   end Generate_Minimal_Vector;

   --  ===========================================
   --  Update Stats
   --  ===========================================

   procedure Update_Stats (
      Stats    : in out Vector_Stats;
      Category : in Vector_Category) is
   begin
      Stats.Total := Stats.Total + 1;

      case Category is
         when Conformance =>
            if Stats.Conformance < Natural'Last then
               Stats.Conformance := Stats.Conformance + 1;
            end if;
         when Boundary =>
            if Stats.Boundary < Natural'Last then
               Stats.Boundary := Stats.Boundary + 1;
            end if;
         when Error_Cat =>
            if Stats.Error_Cnt < Natural'Last then
               Stats.Error_Cnt := Stats.Error_Cnt + 1;
            end if;
         when Performance =>
            if Stats.Performance < Natural'Last then
               Stats.Performance := Stats.Performance + 1;
            end if;
         when Regression =>
            if Stats.Regression < Natural'Last then
               Stats.Regression := Stats.Regression + 1;
            end if;
         when Random_Cat =>
            if Stats.Random_Cnt < Natural'Last then
               Stats.Random_Cnt := Stats.Random_Cnt + 1;
            end if;
      end case;
   end Update_Stats;

   --  ===========================================
   --  Compute Expected Hash
   --  ===========================================

   function Compute_Expected_Hash (Input : String) return String
   is
      Hash : String (1 .. Hash_Length) := (others => '0');
      --  Simple deterministic hash simulation
      Hex_Chars : constant String := "0123456789abcdef";
      Sum : Natural := 0;
   begin
      --  Simple checksum for demonstration
      for C of Input loop
         Sum := (Sum + Character'Pos (C)) mod 256;
      end loop;

      --  Fill hash with deterministic pattern based on sum
      for I in Hash'Range loop
         Hash (I) := Hex_Chars (((Sum + I) mod 16) + 1);
      end loop;

      return Hash;
   end Compute_Expected_Hash;

end Data_Generator;
