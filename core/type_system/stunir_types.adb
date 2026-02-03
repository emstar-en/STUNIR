-------------------------------------------------------------------------------
--  STUNIR Type System - Ada SPARK Body
--  Part of Phase 1 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Stunir_Types is

   -------------------------------------------------------------------------
   --  Make_Type_Name: Create a bounded type name from a string
   -------------------------------------------------------------------------
   function Make_Type_Name (S : String) return Type_Name is
      Result : Type_Name;
   begin
      Result.Length := S'Length;
      for I in S'Range loop
         Result.Data (I - S'First + 1) := S (I);
      end loop;
      return Result;
   end Make_Type_Name;

   -------------------------------------------------------------------------
   --  Equal: Compare two type names
   -------------------------------------------------------------------------
   function Equal (Left, Right : Type_Name) return Boolean is
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
   end Equal;

   -------------------------------------------------------------------------
   --  Make_Int_Type: Create an integer type with specified width and sign
   -------------------------------------------------------------------------
   function Make_Int_Type (Bits : Int_Bits; Signed : Boolean) return STUNIR_Type is
   begin
      return STUNIR_Type'(
         Kind      => Int_Kind,
         Name      => Empty_Type_Name,
         Int_Width => Bits,
         Is_Signed => Signed
      );
   end Make_Int_Type;

   -------------------------------------------------------------------------
   --  Is_Valid: Check if a type is valid (always true by construction)
   -------------------------------------------------------------------------
   function Is_Valid (T : STUNIR_Type) return Boolean is
      pragma Unreferenced (T);
   begin
      return True;
   end Is_Valid;

   -------------------------------------------------------------------------
   --  Size_In_Bits: Get the size of a primitive type in bits
   -------------------------------------------------------------------------
   function Size_In_Bits (T : STUNIR_Type) return Natural is
   begin
      case T.Kind is
         when Int_Kind =>
            return Natural (T.Int_Width);
         when Float_Kind =>
            return Natural (T.Float_Width);
         when Bool_Kind =>
            return 1;
         when Char_Kind =>
            if T.Is_Unicode then
               return 32;  --  Unicode code point
            else
               return 8;   --  ASCII
            end if;
         when others =>
            return 0;  --  Cannot happen due to precondition
      end case;
   end Size_In_Bits;

   -------------------------------------------------------------------------
   --  Types_Equal: Check if two types are equal
   -------------------------------------------------------------------------
   function Types_Equal (Left, Right : STUNIR_Type) return Boolean is
   begin
      if Left.Kind /= Right.Kind then
         return False;
      end if;

      case Left.Kind is
         when Int_Kind =>
            return Left.Int_Width = Right.Int_Width and
                   Left.Is_Signed = Right.Is_Signed;

         when Float_Kind =>
            return Left.Float_Width = Right.Float_Width;

         when Char_Kind =>
            return Left.Is_Unicode = Right.Is_Unicode;

         when String_Kind =>
            return Left.Is_Owned = Right.Is_Owned;

         when Pointer_Kind =>
            return Left.Pointee_Kind = Right.Pointee_Kind and
                   Left.Ptr_Mutable = Right.Ptr_Mutable and
                   Left.Is_Nullable = Right.Is_Nullable;

         when Reference_Kind =>
            return Left.Ref_Kind = Right.Ref_Kind and
                   Left.Ref_Mutable = Right.Ref_Mutable;

         when Array_Kind =>
            return Left.Element_Kind = Right.Element_Kind and
                   Left.Array_Size = Right.Array_Size;

         when Slice_Kind =>
            return Left.Slice_Element = Right.Slice_Element and
                   Left.Slice_Mutable = Right.Slice_Mutable;

         when Struct_Kind =>
            return Equal (Left.Struct_Name, Right.Struct_Name);

         when Union_Kind =>
            return Equal (Left.Union_Name, Right.Union_Name);

         when Enum_Kind =>
            return Equal (Left.Enum_Name, Right.Enum_Name);

         when Tagged_Union_Kind =>
            return Equal (Left.TU_Name, Right.TU_Name);

         when Function_Kind =>
            return Left.Param_Count = Right.Param_Count and
                   Left.Is_Variadic = Right.Is_Variadic and
                   Left.Return_Kind = Right.Return_Kind;

         when Closure_Kind =>
            return Left.Closure_Params = Right.Closure_Params and
                   Left.Closure_Returns = Right.Closure_Returns;

         when Generic_Kind =>
            return Equal (Left.Generic_Base, Right.Generic_Base) and
                   Left.Arg_Count = Right.Arg_Count;

         when Type_Var_Kind =>
            return Equal (Left.Var_Name, Right.Var_Name);

         when Opaque_Kind =>
            return Equal (Left.Opaque_Name, Right.Opaque_Name);

         when Recursive_Kind =>
            return Equal (Left.Rec_Name, Right.Rec_Name);

         when Optional_Kind =>
            return Left.Inner_Kind = Right.Inner_Kind;

         when Result_Kind =>
            return Left.Ok_Kind = Right.Ok_Kind and
                   Left.Err_Kind = Right.Err_Kind;

         when Tuple_Kind =>
            return Left.Tuple_Size = Right.Tuple_Size;

         when Void_Kind | Bool_Kind | Unit_Kind =>
            return True;
      end case;
   end Types_Equal;

   -------------------------------------------------------------------------
   --  Is_Compatible: Check if source type is compatible with target type
   -------------------------------------------------------------------------
   function Is_Compatible (Source, Target : STUNIR_Type) return Boolean is
   begin
      --  Same types are always compatible
      if Types_Equal (Source, Target) then
         return True;
      end if;

      --  Numeric promotions
      if Source.Kind = Int_Kind and Target.Kind = Int_Kind then
         --  Smaller signed can promote to larger signed
         if Source.Is_Signed and Target.Is_Signed then
            return Source.Int_Width <= Target.Int_Width;
         end if;
         --  Unsigned can promote to larger signed
         if not Source.Is_Signed and Target.Is_Signed then
            return Natural (Source.Int_Width) < Natural (Target.Int_Width);
         end if;
         --  Same-size unsigned to unsigned is ok
         if not Source.Is_Signed and not Target.Is_Signed then
            return Source.Int_Width <= Target.Int_Width;
         end if;
      end if;

      --  Int to float promotion
      if Source.Kind = Int_Kind and Target.Kind = Float_Kind then
         return True;  --  Any int can promote to any float
      end if;

      --  Float widening
      if Source.Kind = Float_Kind and Target.Kind = Float_Kind then
         return Source.Float_Width <= Target.Float_Width;
      end if;

      --  Pointer/reference compatibility
      if Source.Kind = Pointer_Kind and Target.Kind = Pointer_Kind then
         --  Void pointer accepts any pointer
         if Target.Pointee_Kind = Void_Kind then
            return True;
         end if;
         --  Same pointee types (mutable can convert to immutable)
         if Source.Pointee_Kind = Target.Pointee_Kind then
            return Source.Ptr_Mutable = Mutable or
                   Target.Ptr_Mutable = Immutable;
         end if;
      end if;

      --  Array to slice conversion
      if Source.Kind = Array_Kind and Target.Kind = Slice_Kind then
         return Source.Element_Kind = Target.Slice_Element;
      end if;

      --  Optional wrapping
      if Target.Kind = Optional_Kind then
         return Source.Kind = Target.Inner_Kind;
      end if;

      return False;
   end Is_Compatible;

   -------------------------------------------------------------------------
   --  Get_Type_Name: Get human-readable type name string
   -------------------------------------------------------------------------
   function Get_Type_Name (T : STUNIR_Type) return String is
   begin
      case T.Kind is
         when Void_Kind =>
            return "void";
         when Unit_Kind =>
            return "()";
         when Bool_Kind =>
            return "bool";
         when Int_Kind =>
            declare
               Prefix : constant String := (if T.Is_Signed then "i" else "u");
               Width_Str : constant String := Int_Bits'Image (T.Int_Width);
            begin
               return Prefix & Width_Str (2 .. Width_Str'Last);
            end;
         when Float_Kind =>
            declare
               Width_Str : constant String := Float_Bits'Image (T.Float_Width);
            begin
               return "f" & Width_Str (2 .. Width_Str'Last);
            end;
         when Char_Kind =>
            return "char";
         when String_Kind =>
            return (if T.Is_Owned then "String" else "&str");
         when Pointer_Kind =>
            return "*ptr";
         when Reference_Kind =>
            return "&ref";
         when Array_Kind =>
            return "[array]";
         when Slice_Kind =>
            return "&[slice]";
         when Struct_Kind =>
            return "struct";
         when Union_Kind =>
            return "union";
         when Enum_Kind =>
            return "enum";
         when Tagged_Union_Kind =>
            return "tagged_union";
         when Function_Kind =>
            return "fn";
         when Closure_Kind =>
            return "closure";
         when Generic_Kind =>
            return "generic";
         when Type_Var_Kind =>
            return "type_var";
         when Opaque_Kind =>
            return "opaque";
         when Recursive_Kind =>
            return "recursive";
         when Optional_Kind =>
            return "Option";
         when Result_Kind =>
            return "Result";
         when Tuple_Kind =>
            return "tuple";
      end case;
   end Get_Type_Name;

   -------------------------------------------------------------------------
   --  Get_IR_Type_String: Get IR representation of type
   -------------------------------------------------------------------------
   function Get_IR_Type_String (T : STUNIR_Type) return String is
   begin
      case T.Kind is
         when Int_Kind =>
            declare
               Prefix : constant String := (if T.Is_Signed then "i" else "u");
               Width_Str : constant String := Int_Bits'Image (T.Int_Width);
            begin
               return Prefix & Width_Str (2 .. Width_Str'Last);
            end;
         when Float_Kind =>
            declare
               Width_Str : constant String := Float_Bits'Image (T.Float_Width);
            begin
               return "f" & Width_Str (2 .. Width_Str'Last);
            end;
         when others =>
            return Get_Type_Name (T);
      end case;
   end Get_IR_Type_String;

end Stunir_Types;
