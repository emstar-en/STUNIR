-------------------------------------------------------------------------------
--  STUNIR Semantic Analysis - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides semantic analysis including dead code detection,
--  unreachable code detection, and constant expression evaluation.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package Semantic_Analysis is

   --  Maximum bounds
   Max_Issues       : constant := 1_000;
   Max_Variables    : constant := 10_000;
   Max_Functions    : constant := 5_000;
   Max_Name_Length  : constant := 256;

   --  Check kinds enumeration
   type Check_Kind is (
      Type_Compatibility,
      Null_Safety,
      Bounds_Check,
      Overflow_Check,
      Dead_Code,
      Unreachable_Code,
      Resource_Leak,
      Constant_Expression,
      Purity,
      Side_Effects
   );

   --  Warning severity levels
   type Warning_Severity is (Info, Warning, Error, Fatal);

   --  Analysis error kinds
   type Analysis_Error_Kind is (
      Undefined_Variable,
      Undefined_Function,
      Undefined_Type,
      Type_Mismatch,
      Dead_Code_Error,
      Unreachable_Code_Error,
      Unused_Variable,
      Unused_Function,
      Invalid_Assignment,
      Invalid_Return
   );

   --  Bounded name type
   subtype Name_Length is Natural range 0 .. Max_Name_Length;
   type Bounded_Name is record
      Data   : String (1 .. Max_Name_Length) := (others => ' ');
      Length : Name_Length := 0;
   end record;

   Empty_Name : constant Bounded_Name := (Data => (others => ' '), Length => 0);

   --  Create a bounded name from a string
   function Make_Name (S : String) return Bounded_Name
     with
       Pre  => S'Length <= Max_Name_Length,
       Post => Make_Name'Result.Length = S'Length;

   --  Semantic issue record
   type Semantic_Issue is record
      Kind     : Analysis_Error_Kind := Undefined_Variable;
      Severity : Warning_Severity := Info;
      Name     : Bounded_Name := Empty_Name;
      Line     : Natural := 0;
      Column   : Natural := 0;
   end record;

   --  Check result record
   type Check_Result is record
      Kind     : Check_Kind := Dead_Code;
      Passed   : Boolean := True;
      Severity : Warning_Severity := Info;
      Line     : Natural := 0;
      Details  : Natural := 0;  --  Context-specific detail
   end record;

   --  Arrays for issues and results
   type Issue_Array is array (Positive range <>) of Semantic_Issue;
   subtype Issue_Vector is Issue_Array (1 .. Max_Issues);

   type Result_Array is array (Positive range <>) of Check_Result;
   subtype Result_Vector is Result_Array (1 .. Max_Issues);

   --  Statement kind for analysis
   type Statement_Kind is (
      Return_Stmt,
      Break_Stmt,
      Continue_Stmt,
      Goto_Stmt,
      Throw_Stmt,
      If_Stmt,
      While_Stmt,
      For_Stmt,
      Var_Decl_Stmt,
      Assign_Stmt,
      Call_Stmt,
      Other_Stmt
   );

   --  Check if a statement kind is terminating
   function Is_Terminating (Kind : Statement_Kind) return Boolean is
     (Kind in Return_Stmt | Break_Stmt | Continue_Stmt | Goto_Stmt | Throw_Stmt);

   --  Variable usage tracking
   type Variable_Info is record
      Name      : Bounded_Name := Empty_Name;
      Assigned  : Natural := 0;
      Used      : Natural := 0;
      Is_Active : Boolean := False;
   end record;

   type Variable_Array is array (Positive range <>) of Variable_Info;
   subtype Variable_Vector is Variable_Array (1 .. Max_Variables);

   --  Function usage tracking
   type Function_Info is record
      Name      : Bounded_Name := Empty_Name;
      Called    : Natural := 0;
      Is_Active : Boolean := False;
   end record;

   type Function_Array is array (Positive range <>) of Function_Info;
   subtype Function_Vector is Function_Array (1 .. Max_Functions);

   --  Dead Code Detector
   type Dead_Code_Detector is record
      Variables      : Variable_Vector := (others => (others => <>));
      Variable_Count : Natural := 0;
      Functions      : Function_Vector := (others => (others => <>));
      Function_Count : Natural := 0;
   end record;

   --  Initialize a dead code detector
   procedure Initialize (Detector : out Dead_Code_Detector);

   --  Register variable assignment
   procedure Register_Assignment (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
     with
       Pre => Name.Length > 0;

   --  Register variable usage
   procedure Register_Usage (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
     with
       Pre => Name.Length > 0;

   --  Register function call
   procedure Register_Call (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
     with
       Pre => Name.Length > 0;

   --  Register function definition
   procedure Register_Function (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
     with
       Pre => Name.Length > 0;

   --  Get dead code results
   procedure Get_Dead_Code_Results (
      Detector : Dead_Code_Detector;
      Results  : out Result_Vector;
      Count    : out Natural)
     with
       Post => Count <= Max_Issues;

   --  Unreachable Code Detector
   type Unreachable_Code_Detector is record
      Results : Result_Vector := (others => (others => <>));
      Count   : Natural := 0;
   end record;

   --  Initialize unreachable code detector
   procedure Initialize (Detector : out Unreachable_Code_Detector);

   --  Add unreachable code result
   procedure Add_Unreachable (
      Detector : in out Unreachable_Code_Detector;
      Line     : Natural;
      Severity : Warning_Severity := Warning)
     with
       Pre => Detector.Count < Max_Issues;

   --  Constant Expression Evaluator
   type Eval_Result_Kind is (Eval_Ok, Eval_Error, Eval_Non_Const);

   type Eval_Result is record
      Kind       : Eval_Result_Kind := Eval_Non_Const;
      Int_Value  : Long_Long_Integer := 0;
      Bool_Value : Boolean := False;
   end record;

   No_Eval_Result : constant Eval_Result := (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);

   --  Evaluate binary integer operation
   function Eval_Binary_Int (
      Op    : Character;
      Left  : Long_Long_Integer;
      Right : Long_Long_Integer) return Eval_Result;

   --  Evaluate comparison operation
   function Eval_Compare (
      Op    : Character;
      Left  : Long_Long_Integer;
      Right : Long_Long_Integer) return Eval_Result;

   --  Evaluate boolean operation
   function Eval_Bool (
      Op    : Character;
      Left  : Boolean;
      Right : Boolean) return Eval_Result;

   --  Semantic Checker (combined)
   type Semantic_Checker is record
      Dead_Detector       : Dead_Code_Detector;
      Unreachable_Detector: Unreachable_Code_Detector;
      Issues              : Issue_Vector := (others => (others => <>));
      Issue_Count         : Natural := 0;
      Has_Errors          : Boolean := False;
   end record;

   --  Initialize semantic checker
   procedure Initialize (Checker : out Semantic_Checker);

   --  Add an issue
   procedure Add_Issue (
      Checker  : in out Semantic_Checker;
      Kind     : Analysis_Error_Kind;
      Severity : Warning_Severity;
      Name     : Bounded_Name;
      Line     : Natural := 0)
     with
       Pre => Checker.Issue_Count < Max_Issues;

   --  Get issue count
   function Get_Issue_Count (Checker : Semantic_Checker) return Natural is
     (Checker.Issue_Count);

   --  Check for errors
   function Has_Errors (Checker : Semantic_Checker) return Boolean is
     (Checker.Has_Errors);

   --  Get summary counts
   type Summary_Counts is record
      Fatal   : Natural := 0;
      Errors  : Natural := 0;
      Warnings: Natural := 0;
      Info    : Natural := 0;
   end record;

   function Get_Summary (Checker : Semantic_Checker) return Summary_Counts;

end Semantic_Analysis;
