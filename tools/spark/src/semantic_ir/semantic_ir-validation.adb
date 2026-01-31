-- STUNIR Semantic IR Validation Package Body
-- DO-178C Level A Compliant
-- SPARK 2014 Mode

pragma SPARK_Mode (On);

package body Semantic_IR.Validation is

   function Make_Error_Result (Msg : String) return Validation_Result is
      Result : Validation_Result;
   begin
      Result.Status := Invalid;
      if Msg'Length <= Max_Error_Length then
         Result.Message := Error_Strings.To_Bounded_String (Msg);
      else
         Result.Message := Error_Strings.To_Bounded_String (
            Msg (Msg'First .. Msg'First + Max_Error_Length - 1)
         );
      end if;
      return Result;
   end Make_Error_Result;
   
   function Make_Warning_Result (Msg : String) return Validation_Result is
      Result : Validation_Result;
   begin
      Result.Status := Warning;
      if Msg'Length <= Max_Error_Length then
         Result.Message := Error_Strings.To_Bounded_String (Msg);
      else
         Result.Message := Error_Strings.To_Bounded_String (
            Msg (Msg'First .. Msg'First + Max_Error_Length - 1)
         );
      end if;
      return Result;
   end Make_Warning_Result;
   
   function Validate_Node_ID (ID : Node_ID) return Validation_Result is
   begin
      if Is_Valid_Node_ID (ID) then
         return Make_Valid_Result;
      else
         return Make_Error_Result ("Invalid node ID: must start with 'n_'");
      end if;
   end Validate_Node_ID;
   
   function Validate_Hash (H : IR_Hash) return Validation_Result is
   begin
      if Is_Valid_Hash (H) then
         return Make_Valid_Result;
      else
         return Make_Error_Result ("Invalid hash: must be 'sha256:' + 64 hex chars");
      end if;
   end Validate_Hash;
   
   function Validate_Type_Reference (T : Type_Reference) return Validation_Result is
   begin
      case T.Kind is
         when TK_Primitive =>
            return Make_Valid_Result;
         when TK_Ref =>
            if Name_Strings.Length (T.Type_Name) > 0 then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Type reference must have non-empty name");
            end if;
         when others =>
            return Make_Warning_Result ("Complex type validation not fully implemented");
      end case;
   end Validate_Type_Reference;
   
   function Validate_Module (M : IR_Module) return Validation_Result is
   begin
      if not Is_Valid_Module (M) then
         return Make_Error_Result ("Module structure is invalid");
      end if;
      
      -- Validate node ID
      declare
         ID_Result : constant Validation_Result := Validate_Node_ID (M.Base.Node_ID);
      begin
         if ID_Result.Status /= Valid then
            return ID_Result;
         end if;
      end;
      
      -- Validate hash
      declare
         Hash_Result : constant Validation_Result := Validate_Hash (M.Base.Hash);
      begin
         if Hash_Result.Status /= Valid then
            return Hash_Result;
         end if;
      end;
      
      return Make_Valid_Result;
   end Validate_Module;
   
   function Check_Type_Compatibility (
      T1 : Type_Reference;
      T2 : Type_Reference
   ) return Boolean is
   begin
      -- Simplified type compatibility check
      if T1.Kind = TK_Primitive and then T2.Kind = TK_Primitive then
         return T1.Prim_Type = T2.Prim_Type;
      elsif T1.Kind = TK_Ref and then T2.Kind = TK_Ref then
         return Name_Strings."="(T1.Type_Name, T2.Type_Name);
      else
         return False;
      end if;
   end Check_Type_Compatibility;
   
   function Check_Binary_Op_Types (
      Op    : Binary_Operator;
      Left  : Type_Reference;
      Right : Type_Reference
   ) return Validation_Result is
   begin
      -- Simplified type checking for binary operators
      if not (Left.Kind = TK_Primitive and then Right.Kind = TK_Primitive) then
         return Make_Error_Result ("Binary operators require primitive types");
      end if;
      
      case Op is
         when Op_Add | Op_Sub | Op_Mul | Op_Div | Op_Mod =>
            -- Arithmetic operators need numeric types
            if Left.Prim_Type in Type_I8 .. Type_F64 and then
               Right.Prim_Type in Type_I8 .. Type_F64 then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Arithmetic operators require numeric types");
            end if;
            
         when Op_Eq | Op_Neq | Op_Lt | Op_Leq | Op_Gt | Op_Geq =>
            -- Comparison operators need compatible types
            if Check_Type_Compatibility (Left, Right) then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Comparison operators require compatible types");
            end if;
            
         when Op_And | Op_Or =>
            -- Logical operators need boolean types
            if Left.Prim_Type = Type_Bool and then Right.Prim_Type = Type_Bool then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Logical operators require boolean types");
            end if;
            
         when Op_Bit_And | Op_Bit_Or | Op_Bit_Xor | Op_Shl | Op_Shr =>
            -- Bitwise operators need integer types
            if Left.Prim_Type in Type_I8 .. Type_U64 and then
               Right.Prim_Type in Type_I8 .. Type_U64 then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Bitwise operators require integer types");
            end if;
            
         when Op_Assign =>
            -- Assignment requires compatible types
            if Check_Type_Compatibility (Left, Right) then
               return Make_Valid_Result;
            else
               return Make_Error_Result ("Assignment requires compatible types");
            end if;
      end case;
   end Check_Binary_Op_Types;
   
end Semantic_IR.Validation;
