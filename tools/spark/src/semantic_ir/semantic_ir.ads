-- STUNIR Semantic IR Root Package
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package Semantic_IR with
   Pure,
   SPARK_Mode => On
is
   -- Root package for Semantic IR hierarchy
   -- Version information
   IR_Version_Major : constant := 1;
   IR_Version_Minor : constant := 0;
   IR_Version_Patch : constant := 0;
   
   Schema_Version : constant String := "1.0.0";
   
end Semantic_IR;
