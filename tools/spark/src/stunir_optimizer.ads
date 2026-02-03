--  STUNIR IR Optimizer - Ada SPARK specification (v0.8.9+)
--
--  Provides optimization passes for STUNIR IR:
--  - Dead code elimination
--  - Constant folding
--  - Unreachable code elimination
--
--  DO-178C Level A compliant implementation.

with Ada.Strings.Bounded;

package STUNIR_Optimizer
   with SPARK_Mode => On
is
   --  Version constant
   Version : constant String := "0.8.9";

   --  Maximum optimization iterations
   Max_Iterations : constant := 10;

   --  Optimization levels
   type Optimization_Level is (O0, O1, O2, O3);
   --  O0 = No optimization
   --  O1 = Basic (dead code elimination, constant folding)
   --  O2 = Standard (O1 + unreachable code elimination)
   --  O3 = Aggressive (all passes)

   --  Optimization pass types
   type Optimization_Pass is
      (Dead_Code_Elimination,
       Constant_Folding,
       Constant_Propagation,
       Unreachable_Code_Elimination);

   --  Optimization result
   type Optimization_Result is record
      Success      : Boolean;
      Changes      : Natural;
      Iterations   : Natural;
   end record;

   --  String buffers for IR content
   package Content_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 65536);
   subtype Content_String is Content_Strings.Bounded_String;

   --  Initialize optimizer with given level
   procedure Initialize (Level : Optimization_Level)
      with Global => null;

   --  Run dead code elimination pass
   function Eliminate_Dead_Code
      (IR_Content : Content_String) return Content_String
      with Pre => Content_Strings.Length (IR_Content) > 0;

   --  Run constant folding pass
   function Fold_Constants
      (IR_Content : Content_String) return Content_String
      with Pre => Content_Strings.Length (IR_Content) > 0;

   --  Run unreachable code elimination pass
   function Eliminate_Unreachable
      (IR_Content : Content_String) return Content_String
      with Pre => Content_Strings.Length (IR_Content) > 0;

   --  Run all optimization passes based on level
   function Optimize_IR
      (IR_Content : Content_String;
       Level      : Optimization_Level) return Optimization_Result
      with Pre => Content_Strings.Length (IR_Content) > 0;

   --  Try to fold a constant expression
   --  Returns True if folding was successful, with result in Folded_Value
   procedure Try_Fold_Expression
      (Expr         : in String;
       Folded_Value : out Integer;
       Success      : out Boolean)
      with Pre => Expr'Length > 0;

   --  Check if an expression is a constant boolean
   function Is_Constant_Boolean (Expr : String) return Boolean
      with Pre => Expr'Length > 0;

   --  Check if an expression evaluates to true
   function Is_Constant_True (Expr : String) return Boolean
      with Pre => Expr'Length > 0;

   --  Check if an expression evaluates to false
   function Is_Constant_False (Expr : String) return Boolean
      with Pre => Expr'Length > 0;

end STUNIR_Optimizer;
