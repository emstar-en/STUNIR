-- STUNIR Semantic IR Validation Package Specification
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

with Semantic_IR.Types; use Semantic_IR.Types;
with Semantic_IR.Nodes; use Semantic_IR.Nodes;
with Semantic_IR.Modules; use Semantic_IR.Modules;

package Semantic_IR.Validation with
   SPARK_Mode => On
is
   -- Validation result
   type Validation_Status is (Valid, Invalid, Warning);
   
   Max_Error_Length : constant := 512;
   package Error_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max_Error_Length);
   subtype Error_Message is Error_Strings.Bounded_String;
   
   type Validation_Result is record
      Status  : Validation_Status := Valid;
      Message : Error_Message;
   end record;
   
   -- Validation functions
   function Validate_Node_ID (ID : Node_ID) return Validation_Result
      with Post => (if Validate_Node_ID'Result.Status = Valid then
                       Is_Valid_Node_ID (ID));
   
   function Validate_Hash (H : IR_Hash) return Validation_Result
      with Post => (if Validate_Hash'Result.Status = Valid then
                       Is_Valid_Hash (H));
   
   function Validate_Type_Reference (T : Type_Reference) return Validation_Result;
   
   function Validate_Module (M : IR_Module) return Validation_Result
      with Post => (if Validate_Module'Result.Status = Valid then
                       Is_Valid_Module (M));
   
   -- Semantic checks
   function Check_Type_Compatibility (
      T1 : Type_Reference;
      T2 : Type_Reference
   ) return Boolean;
   
   function Check_Binary_Op_Types (
      Op    : Binary_Operator;
      Left  : Type_Reference;
      Right : Type_Reference
   ) return Validation_Result;
   
   -- Helper functions
   function Make_Valid_Result return Validation_Result is
      (Validation_Result'(Status => Valid, Message => Error_Strings.Null_Bounded_String));
   
   function Make_Error_Result (Msg : String) return Validation_Result;
   
   function Make_Warning_Result (Msg : String) return Validation_Result;
   
end Semantic_IR.Validation;
