-------------------------------------------------------------------------------
--  STUNIR Semantic Analysis Tests - Ada
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Semantic_Analysis; use Semantic_Analysis;

procedure Test_Semantic is
   Detector : Dead_Code_Detector;
   Results  : Result_Vector;
   Count    : Natural;
   Checker  : Semantic_Checker;
   Summary  : Summary_Counts;
   
   Test_Count : Natural := 0;
   Pass_Count : Natural := 0;
   
   procedure Test (Name : String; Passed : Boolean) is
   begin
      Test_Count := Test_Count + 1;
      if Passed then
         Pass_Count := Pass_Count + 1;
         Put_Line ("  ✓ " & Name);
      else
         Put_Line ("  ✗ " & Name);
      end if;
   end Test;
   
begin
   Put_Line ("STUNIR Semantic Analysis Tests");
   Put_Line ("==============================");
   Put_Line ("");
   
   -- Test Dead Code Detector
   Put_Line ("Dead Code Detector Tests:");
   
   Initialize (Detector);
   
   -- Register variables and track usage
   Register_Assignment (Detector, Make_Name ("x"));
   Register_Assignment (Detector, Make_Name ("y"));
   Register_Assignment (Detector, Make_Name ("z"));
   
   Register_Usage (Detector, Make_Name ("x"));
   Register_Usage (Detector, Make_Name ("y"));
   -- z is not used -> dead code
   
   Get_Dead_Code_Results (Detector, Results, Count);
   Test ("Detects unused variable", Count > 0);
   
   -- Reset and test functions
   Initialize (Detector);
   
   Register_Function (Detector, Make_Name ("main"));
   Register_Function (Detector, Make_Name ("helper"));
   Register_Function (Detector, Make_Name ("unused_func"));
   
   Register_Call (Detector, Make_Name ("helper"));
   -- unused_func is not called -> dead code (main is exempt)
   
   Get_Dead_Code_Results (Detector, Results, Count);
   Test ("Detects unused function", Count >= 1);
   
   -- Test Constant Evaluation
   Put_Line ("");
   Put_Line ("Constant Evaluation Tests:");
   
   declare
      R : Eval_Result;
   begin
      R := Eval_Binary_Int ('+', 10, 20);
      Test ("10 + 20 = 30", R.Kind = Eval_Ok and R.Int_Value = 30);
      
      R := Eval_Binary_Int ('-', 50, 20);
      Test ("50 - 20 = 30", R.Kind = Eval_Ok and R.Int_Value = 30);
      
      R := Eval_Binary_Int ('*', 6, 7);
      Test ("6 * 7 = 42", R.Kind = Eval_Ok and R.Int_Value = 42);
      
      R := Eval_Binary_Int ('/', 100, 4);
      Test ("100 / 4 = 25", R.Kind = Eval_Ok and R.Int_Value = 25);
      
      R := Eval_Binary_Int ('/', 100, 0);
      Test ("Division by zero returns error", R.Kind = Eval_Error);
      
      R := Eval_Compare ('=', 42, 42);
      Test ("42 = 42 is true", R.Kind = Eval_Ok and R.Bool_Value);
      
      R := Eval_Compare ('<', 10, 20);
      Test ("10 < 20 is true", R.Kind = Eval_Ok and R.Bool_Value);
      
      R := Eval_Bool ('&', True, True);
      Test ("true && true = true", R.Kind = Eval_Ok and R.Bool_Value);
      
      R := Eval_Bool ('|', False, True);
      Test ("false || true = true", R.Kind = Eval_Ok and R.Bool_Value);
   end;
   
   -- Test Semantic Checker
   Put_Line ("");
   Put_Line ("Semantic Checker Tests:");
   
   Initialize (Checker);
   Test ("Checker initialized", Get_Issue_Count (Checker) = 0);
   Test ("No initial errors", not Has_Errors (Checker));
   
   Add_Issue (Checker, Unused_Variable, Warning, Make_Name ("test_var"), 10);
   Test ("Add warning issue", Get_Issue_Count (Checker) = 1);
   Test ("Warning doesn't set has_errors", not Has_Errors (Checker));
   
   Add_Issue (Checker, Type_Mismatch, Error, Make_Name ("x"), 20);
   Test ("Add error issue", Get_Issue_Count (Checker) = 2);
   Test ("Error sets has_errors", Has_Errors (Checker));
   
   Summary := Get_Summary (Checker);
   Test ("Summary has 1 warning", Summary.Warnings = 1);
   Test ("Summary has 1 error", Summary.Errors = 1);
   
   -- Test statement termination
   Put_Line ("");
   Put_Line ("Statement Termination Tests:");
   
   Test ("Return is terminating", Is_Terminating (Return_Stmt));
   Test ("Break is terminating", Is_Terminating (Break_Stmt));
   Test ("Continue is terminating", Is_Terminating (Continue_Stmt));
   Test ("Goto is terminating", Is_Terminating (Goto_Stmt));
   Test ("If is not terminating", not Is_Terminating (If_Stmt));
   Test ("While is not terminating", not Is_Terminating (While_Stmt));
   Test ("Assign is not terminating", not Is_Terminating (Assign_Stmt));
   
   -- Summary
   Put_Line ("");
   Put_Line ("===================");
   Put_Line ("Results:" & Natural'Image (Pass_Count) & " /" & Natural'Image (Test_Count) & " passed");
   
   if Pass_Count = Test_Count then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED.");
   end if;
   
end Test_Semantic;
