--  STUNIR DO-332 Substitutability Checker Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Inheritance_Analyzer; use Inheritance_Analyzer;

package body Substitutability is

   --  ============================================================
   --  Check_LSP Implementation
   --  ============================================================

   function Check_LSP (
      Parent_Method : Method_Info;
      Child_Method  : Method_Info
   ) return LSP_Check_Result is
   begin
      --  Check signature compatibility first
      if not Check_Signature_Compatible (Parent_Method, Child_Method) then
         return LSP_Signature_Mismatch;
      end if;

      --  Check covariance of return type
      if not Child_Method.Is_Covariant then
         return LSP_Covariance_Violation;
      end if;

      --  Check contravariance of parameters
      if Child_Method.Is_Contravariant then
         --  Contravariance flag set but may indicate violation
         return LSP_Contravariance_Violation;
      end if;

      return LSP_Compliant;
   end Check_LSP;

   --  ============================================================
   --  Is_Substitutable Implementation
   --  ============================================================

   function Is_Substitutable (
      Parent_Class : Class_ID;
      Child_Class  : Class_ID;
      Methods      : Method_Array;
      Links        : Inheritance_Array
   ) return Boolean is
      Check_Result : LSP_Check_Result;
   begin
      --  Check all overriding methods
      for I in Methods'Range loop
         if Methods (I).Owning_Class = Child_Class and Methods (I).Has_Override then
            --  Find parent method
            for J in Methods'Range loop
               if Methods (J).ID = Methods (I).Override_Of and
                  Methods (J).Owning_Class = Parent_Class then
                  Check_Result := Check_LSP (Methods (J), Methods (I));
                  if Check_Result /= LSP_Compliant then
                     return False;
                  end if;
                  exit;
               end if;
            end loop;
         end if;
      end loop;

      return True;
   end Is_Substitutable;

   --  ============================================================
   --  Find_LSP_Violations Implementation
   --  ============================================================

   procedure Find_LSP_Violations (
      Classes    : in     Class_Array;
      Methods    : in     Method_Array;
      Links      : in     Inheritance_Array;
      Summary    :    out Substitutability_Summary;
      Has_Errors :    out Boolean
   ) is
      Check_Result : LSP_Check_Result;
      Total        : Natural := 0;
      Compliant    : Natural := 0;
   begin
      Summary := Null_Summary;
      Has_Errors := False;

      --  Check all method overrides
      for I in Methods'Range loop
         if Methods (I).Has_Override then
            Total := Total + 1;

            --  Find the parent method
            for J in Methods'Range loop
               if Methods (J).ID = Methods (I).Override_Of then
                  Check_Result := Check_LSP (Methods (J), Methods (I));

                  case Check_Result is
                     when LSP_Compliant =>
                        Compliant := Compliant + 1;

                     when LSP_Precondition_Strengthened =>
                        Summary.Precond_Violations := Summary.Precond_Violations + 1;
                        Has_Errors := True;

                     when LSP_Postcondition_Weakened =>
                        Summary.Postcond_Violations := Summary.Postcond_Violations + 1;
                        Has_Errors := True;

                     when LSP_Invariant_Violated =>
                        Summary.Invariant_Violations := Summary.Invariant_Violations + 1;
                        Has_Errors := True;

                     when LSP_Covariance_Violation =>
                        Summary.Covariance_Violations := Summary.Covariance_Violations + 1;
                        Has_Errors := True;

                     when LSP_Contravariance_Violation =>
                        Summary.Contravariance_Violations := Summary.Contravariance_Violations + 1;
                        Has_Errors := True;

                     when LSP_Exception_Mismatch =>
                        Summary.Exception_Violations := Summary.Exception_Violations + 1;
                        Has_Errors := True;

                     when LSP_Signature_Mismatch =>
                        Summary.Signature_Violations := Summary.Signature_Violations + 1;
                        Has_Errors := True;
                  end case;

                  exit;
               end if;
            end loop;
         end if;
      end loop;

      Summary.Total_Checked := Total;
      Summary.Total_Compliant := Compliant;
   end Find_LSP_Violations;

   --  ============================================================
   --  Check_Signature_Compatible Implementation
   --  ============================================================

   function Check_Signature_Compatible (
      Parent_Method : Method_Info;
      Child_Method  : Method_Info
   ) return Boolean is
   begin
      --  Parameter count must match
      if Parent_Method.Parameter_Count /= Child_Method.Parameter_Count then
         return False;
      end if;

      --  For now, assume compatible if counts match
      --  Full implementation would check parameter types
      return True;
   end Check_Signature_Compatible;

   --  ============================================================
   --  Check_Return_Type_Covariance Implementation
   --  ============================================================

   function Check_Return_Type_Covariance (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info;
      Links         : Inheritance_Array
   ) return Boolean is
   begin
      --  Simplified: use the Is_Covariant flag
      return Child_Method.Is_Covariant;
   end Check_Return_Type_Covariance;

   --  ============================================================
   --  Check_Parameter_Contravariance Implementation
   --  ============================================================

   function Check_Parameter_Contravariance (
      Child_Method  : Method_Info;
      Parent_Method : Method_Info;
      Links         : Inheritance_Array
   ) return Boolean is
   begin
      --  Simplified: contravariant flag should NOT be set for valid LSP
      return not Child_Method.Is_Contravariant;
   end Check_Parameter_Contravariance;

end Substitutability;
