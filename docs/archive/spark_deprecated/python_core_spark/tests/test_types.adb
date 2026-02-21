-------------------------------------------------------------------------------
--  STUNIR Type System Tests - Ada
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stunir_Types; use Stunir_Types;
with Stunir_Type_Registry; use Stunir_Type_Registry;

procedure Test_Types is
   Registry : Type_Registry;
   T : STUNIR_Type;
   Id : Type_Id;
   
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
   Put_Line ("STUNIR Type System Tests");
   Put_Line ("========================");
   Put_Line ("");
   
   -- Test type creation
   Put_Line ("Type Creation Tests:");
   
   T := I32;
   Test ("I32 type creation", T.Kind = Int_Kind and T.Int_Width = 32 and T.Is_Signed);
   
   T := U64;
   Test ("U64 type creation", T.Kind = Int_Kind and T.Int_Width = 64 and not T.Is_Signed);
   
   T := F64;
   Test ("F64 type creation", T.Kind = Float_Kind and T.Float_Width = 64);
   
   T := Bool_Type;
   Test ("Bool type creation", T.Kind = Bool_Kind);
   
   T := Void_Type;
   Test ("Void type creation", T.Kind = Void_Kind);
   
   -- Test type predicates
   Put_Line ("");
   Put_Line ("Type Predicate Tests:");
   
   Test ("I32 is primitive", Is_Primitive (I32));
   Test ("Bool is primitive", Is_Primitive (Bool_Type));
   Test ("I32 is numeric", Is_Numeric (I32));
   Test ("F64 is numeric", Is_Numeric (F64));
   Test ("I32 is integer", Is_Integer (I32));
   Test ("F64 is floating", Is_Floating (F64));
   
   -- Test type equality
   Put_Line ("");
   Put_Line ("Type Equality Tests:");
   
   Test ("I32 equals I32", Types_Equal (I32, I32));
   Test ("I32 not equals U32", not Types_Equal (I32, U32));
   Test ("F32 not equals F64", not Types_Equal (F32, F64));
   
   -- Test type compatibility
   Put_Line ("");
   Put_Line ("Type Compatibility Tests:");
   
   Test ("I16 compatible with I32", Is_Compatible (I16, I32));
   Test ("I32 compatible with I64", Is_Compatible (I32, I64));
   Test ("I32 compatible with F64", Is_Compatible (I32, F64));
   Test ("F32 compatible with F64", Is_Compatible (F32, F64));
   
   -- Test type registry
   Put_Line ("");
   Put_Line ("Type Registry Tests:");
   
   Initialize (Registry);
   Test ("Registry initialized with builtins", Get_Count (Registry) >= 15);
   
   Id := Lookup (Registry, Make_Type_Name ("i32"));
   Test ("Lookup i32 succeeds", Id /= No_Type_Id);
   
   Id := Lookup (Registry, Make_Type_Name ("bool"));
   Test ("Lookup bool succeeds", Id /= No_Type_Id);
   
   Id := Lookup (Registry, Make_Type_Name ("nonexistent"));
   Test ("Lookup nonexistent returns No_Type_Id", Id = No_Type_Id);
   
   -- Test custom type registration
   Register (Registry, Make_Type_Name ("my_type"), I32, Id);
   Test ("Register custom type", Id /= No_Type_Id);
   
   Id := Lookup (Registry, Make_Type_Name ("my_type"));
   Test ("Lookup custom type", Id /= No_Type_Id);
   
   -- Summary
   Put_Line ("");
   Put_Line ("===================");
   Put_Line ("Results:" & Natural'Image (Pass_Count) & " /" & Natural'Image (Test_Count) & " passed");
   
   if Pass_Count = Test_Count then
      Put_Line ("All tests PASSED!");
   else
      Put_Line ("Some tests FAILED.");
   end if;
   
end Test_Types;
