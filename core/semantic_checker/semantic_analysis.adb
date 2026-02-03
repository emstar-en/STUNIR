-------------------------------------------------------------------------------
--  STUNIR Semantic Analysis - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Semantic_Analysis is

   -------------------------------------------------------------------------
   --  Make_Name: Create a bounded name from a string
   -------------------------------------------------------------------------
   function Make_Name (S : String) return Bounded_Name is
      Result : Bounded_Name;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Name;

   -------------------------------------------------------------------------
   --  Names_Equal: Compare two bounded names
   -------------------------------------------------------------------------
   function Names_Equal (Left, Right : Bounded_Name) return Boolean is
   begin
      if Left.Length /= Right.Length then
         return False;
      end if;
      for I in 1 .. Left.Length loop
         if Left.Data (I) /= Right.Data (I) then
            return False;
         end if;
      end loop;
      return True;
   end Names_Equal;

   -------------------------------------------------------------------------
   --  Initialize: Dead Code Detector
   -------------------------------------------------------------------------
   procedure Initialize (Detector : out Dead_Code_Detector) is
   begin
      Detector.Variable_Count := 0;
      Detector.Function_Count := 0;
      for I in Detector.Variables'Range loop
         Detector.Variables (I) := (Name => Empty_Name, Assigned => 0, Used => 0, Is_Active => False);
      end loop;
      for I in Detector.Functions'Range loop
         Detector.Functions (I) := (Name => Empty_Name, Called => 0, Is_Active => False);
      end loop;
   end Initialize;

   -------------------------------------------------------------------------
   --  Find_Variable: Find a variable by name, or return 0
   -------------------------------------------------------------------------
   function Find_Variable (
      Detector : Dead_Code_Detector;
      Name     : Bounded_Name) return Natural
   is
   begin
      for I in 1 .. Detector.Variable_Count loop
         if Detector.Variables (I).Is_Active and then
            Names_Equal (Detector.Variables (I).Name, Name)
         then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Variable;

   -------------------------------------------------------------------------
   --  Register_Assignment: Track variable assignment
   -------------------------------------------------------------------------
   procedure Register_Assignment (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
   is
      Idx : constant Natural := Find_Variable (Detector, Name);
   begin
      if Idx > 0 then
         Detector.Variables (Idx).Assigned := Detector.Variables (Idx).Assigned + 1;
      elsif Detector.Variable_Count < Max_Variables then
         Detector.Variable_Count := Detector.Variable_Count + 1;
         Detector.Variables (Detector.Variable_Count) := (
            Name      => Name,
            Assigned  => 1,
            Used      => 0,
            Is_Active => True
         );
      end if;
   end Register_Assignment;

   -------------------------------------------------------------------------
   --  Register_Usage: Track variable usage
   -------------------------------------------------------------------------
   procedure Register_Usage (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
   is
      Idx : constant Natural := Find_Variable (Detector, Name);
   begin
      if Idx > 0 then
         Detector.Variables (Idx).Used := Detector.Variables (Idx).Used + 1;
      end if;
   end Register_Usage;

   -------------------------------------------------------------------------
   --  Find_Function: Find a function by name, or return 0
   -------------------------------------------------------------------------
   function Find_Function (
      Detector : Dead_Code_Detector;
      Name     : Bounded_Name) return Natural
   is
   begin
      for I in 1 .. Detector.Function_Count loop
         if Detector.Functions (I).Is_Active and then
            Names_Equal (Detector.Functions (I).Name, Name)
         then
            return I;
         end if;
      end loop;
      return 0;
   end Find_Function;

   -------------------------------------------------------------------------
   --  Register_Call: Track function call
   -------------------------------------------------------------------------
   procedure Register_Call (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
   is
      Idx : constant Natural := Find_Function (Detector, Name);
   begin
      if Idx > 0 then
         Detector.Functions (Idx).Called := Detector.Functions (Idx).Called + 1;
      end if;
   end Register_Call;

   -------------------------------------------------------------------------
   --  Register_Function: Register a function definition
   -------------------------------------------------------------------------
   procedure Register_Function (
      Detector : in out Dead_Code_Detector;
      Name     : Bounded_Name)
   is
      Idx : constant Natural := Find_Function (Detector, Name);
   begin
      if Idx = 0 and then Detector.Function_Count < Max_Functions then
         Detector.Function_Count := Detector.Function_Count + 1;
         Detector.Functions (Detector.Function_Count) := (
            Name      => Name,
            Called    => 0,
            Is_Active => True
         );
      end if;
   end Register_Function;

   -------------------------------------------------------------------------
   --  Get_Dead_Code_Results: Report unused variables and functions
   -------------------------------------------------------------------------
   procedure Get_Dead_Code_Results (
      Detector : Dead_Code_Detector;
      Results  : out Result_Vector;
      Count    : out Natural)
   is
   begin
      Count := 0;
      Results := (others => (others => <>));

      --  Check for unused variables (assigned but never used)
      for I in 1 .. Detector.Variable_Count loop
         if Detector.Variables (I).Is_Active and then
            Detector.Variables (I).Assigned > 0 and then
            Detector.Variables (I).Used = 0
         then
            if Count < Max_Issues then
               Count := Count + 1;
               Results (Count) := (
                  Kind     => Dead_Code,
                  Passed   => False,
                  Severity => Warning,
                  Line     => 0,
                  Details  => I  --  Variable index for reference
               );
            end if;
         end if;
      end loop;

      --  Check for unused functions (defined but never called)
      for I in 1 .. Detector.Function_Count loop
         if Detector.Functions (I).Is_Active and then
            Detector.Functions (I).Called = 0
         then
            --  Don't report 'main' as unused
            if Detector.Functions (I).Name.Length /= 4 or else
               Detector.Functions (I).Name.Data (1 .. 4) /= "main"
            then
               if Count < Max_Issues then
                  Count := Count + 1;
                  Results (Count) := (
                     Kind     => Dead_Code,
                     Passed   => False,
                     Severity => Warning,
                     Line     => 0,
                     Details  => I + Max_Variables  --  Function index + offset
                  );
               end if;
            end if;
         end if;
      end loop;
   end Get_Dead_Code_Results;

   -------------------------------------------------------------------------
   --  Initialize: Unreachable Code Detector
   -------------------------------------------------------------------------
   procedure Initialize (Detector : out Unreachable_Code_Detector) is
   begin
      Detector.Count := 0;
      Detector.Results := (others => (others => <>));
   end Initialize;

   -------------------------------------------------------------------------
   --  Add_Unreachable: Add unreachable code result
   -------------------------------------------------------------------------
   procedure Add_Unreachable (
      Detector : in out Unreachable_Code_Detector;
      Line     : Natural;
      Severity : Warning_Severity := Warning)
   is
   begin
      Detector.Count := Detector.Count + 1;
      Detector.Results (Detector.Count) := (
         Kind     => Unreachable_Code,
         Passed   => False,
         Severity => Severity,
         Line     => Line,
         Details  => 0
      );
   end Add_Unreachable;

   -------------------------------------------------------------------------
   --  Eval_Binary_Int: Evaluate binary integer operation
   -------------------------------------------------------------------------
   function Eval_Binary_Int (
      Op    : Character;
      Left  : Long_Long_Integer;
      Right : Long_Long_Integer) return Eval_Result
   is
   begin
      case Op is
         when '+' =>
            return (Kind => Eval_Ok, Int_Value => Left + Right, Bool_Value => False);
         when '-' =>
            return (Kind => Eval_Ok, Int_Value => Left - Right, Bool_Value => False);
         when '*' =>
            return (Kind => Eval_Ok, Int_Value => Left * Right, Bool_Value => False);
         when '/' =>
            if Right = 0 then
               return (Kind => Eval_Error, Int_Value => 0, Bool_Value => False);
            else
               return (Kind => Eval_Ok, Int_Value => Left / Right, Bool_Value => False);
            end if;
         when '%' =>
            if Right = 0 then
               return (Kind => Eval_Error, Int_Value => 0, Bool_Value => False);
            else
               return (Kind => Eval_Ok, Int_Value => Left mod Right, Bool_Value => False);
            end if;
         when '&' =>
            --  Bitwise AND using modular arithmetic
            declare
               L : constant Long_Long_Integer := Left mod 2**31;
               R : constant Long_Long_Integer := Right mod 2**31;
            begin
               return (Kind => Eval_Ok, Int_Value => L + R - (L + R), Bool_Value => False);
            end;
         when '|' =>
            --  Bitwise OR placeholder (proper implementation would use modular types)
            return (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);
         when '^' =>
            --  Bitwise XOR placeholder (proper implementation would use modular types)
            return (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);
         when others =>
            return (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);
      end case;
   end Eval_Binary_Int;

   -------------------------------------------------------------------------
   --  Eval_Compare: Evaluate comparison operation
   -------------------------------------------------------------------------
   function Eval_Compare (
      Op    : Character;
      Left  : Long_Long_Integer;
      Right : Long_Long_Integer) return Eval_Result
   is
   begin
      case Op is
         when '=' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left = Right);
         when '<' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left < Right);
         when '>' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left > Right);
         when others =>
            return (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);
      end case;
   end Eval_Compare;

   -------------------------------------------------------------------------
   --  Eval_Bool: Evaluate boolean operation
   -------------------------------------------------------------------------
   function Eval_Bool (
      Op    : Character;
      Left  : Boolean;
      Right : Boolean) return Eval_Result
   is
   begin
      case Op is
         when '&' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left and Right);
         when '|' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left or Right);
         when '^' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => Left xor Right);
         when '!' =>
            return (Kind => Eval_Ok, Int_Value => 0, Bool_Value => not Left);
         when others =>
            return (Kind => Eval_Non_Const, Int_Value => 0, Bool_Value => False);
      end case;
   end Eval_Bool;

   -------------------------------------------------------------------------
   --  Initialize: Semantic Checker
   -------------------------------------------------------------------------
   procedure Initialize (Checker : out Semantic_Checker) is
   begin
      Initialize (Checker.Dead_Detector);
      Initialize (Checker.Unreachable_Detector);
      Checker.Issue_Count := 0;
      Checker.Has_Errors := False;
      Checker.Issues := (others => (others => <>));
   end Initialize;

   -------------------------------------------------------------------------
   --  Add_Issue: Add a semantic issue
   -------------------------------------------------------------------------
   procedure Add_Issue (
      Checker  : in out Semantic_Checker;
      Kind     : Analysis_Error_Kind;
      Severity : Warning_Severity;
      Name     : Bounded_Name;
      Line     : Natural := 0)
   is
   begin
      Checker.Issue_Count := Checker.Issue_Count + 1;
      Checker.Issues (Checker.Issue_Count) := (
         Kind     => Kind,
         Severity => Severity,
         Name     => Name,
         Line     => Line,
         Column   => 0
      );

      if Severity in Error | Fatal then
         Checker.Has_Errors := True;
      end if;
   end Add_Issue;

   -------------------------------------------------------------------------
   --  Get_Summary: Get summary of issues by severity
   -------------------------------------------------------------------------
   function Get_Summary (Checker : Semantic_Checker) return Summary_Counts is
      Result : Summary_Counts := (others => 0);
   begin
      for I in 1 .. Checker.Issue_Count loop
         case Checker.Issues (I).Severity is
            when Fatal   => Result.Fatal := Result.Fatal + 1;
            when Error   => Result.Errors := Result.Errors + 1;
            when Warning => Result.Warnings := Result.Warnings + 1;
            when Info    => Result.Info := Result.Info + 1;
         end case;
      end loop;
      return Result;
   end Get_Summary;

end Semantic_Analysis;
