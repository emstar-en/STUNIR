-------------------------------------------------------------------------------
--  STUNIR Type System - Ada SPARK Specification
--  Part of Phase 1 SPARK Migration
--
--  This package provides the core type system for STUNIR cross-language
--  code generation with formal verification.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Containers.Formal_Ordered_Maps;
with Ada.Containers.Formal_Vectors;

package Stunir_Types is

   --  Maximum constants for bounded containers
   Max_Type_Name_Length : constant := 256;
   Max_Fields           : constant := 128;
   Max_Generic_Params   : constant := 16;
   Max_Enum_Variants    : constant := 256;

   --  Type kind enumeration (24 variants matching Python)
   type Type_Kind is (
      Void_Kind,
      Bool_Kind,
      Int_Kind,
      Float_Kind,
      Char_Kind,
      String_Kind,
      Pointer_Kind,
      Reference_Kind,
      Array_Kind,
      Slice_Kind,
      Struct_Kind,
      Union_Kind,
      Enum_Kind,
      Tagged_Union_Kind,
      Function_Kind,
      Closure_Kind,
      Generic_Kind,
      Type_Var_Kind,
      Opaque_Kind,
      Recursive_Kind,
      Optional_Kind,
      Result_Kind,
      Tuple_Kind,
      Unit_Kind
   );

   --  Ownership semantics (primarily for Rust)
   type Ownership_Kind is (
      Owned,
      Borrowed,
      Borrowed_Mut,
      Copy_Ownership,
      Static_Ownership
   );

   --  Mutability of references/variables
   type Mutability_Kind is (
      Immutable,
      Mutable,
      Const_Mut
   );

   --  Valid integer bit widths
   type Int_Bits is range 8 .. 128
     with Static_Predicate => Int_Bits in 8 | 16 | 32 | 64 | 128;

   --  Valid float bit widths
   type Float_Bits is range 16 .. 128
     with Static_Predicate => Float_Bits in 16 | 32 | 64 | 128;

   --  Bounded type name string
   subtype Type_Name_Length is Natural range 0 .. Max_Type_Name_Length;
   type Type_Name is record
      Data   : String (1 .. Max_Type_Name_Length) := (others => ' ');
      Length : Type_Name_Length := 0;
   end record;

   --  Empty type name constant
   Empty_Type_Name : constant Type_Name := (Data => (others => ' '), Length => 0);

   --  Create a type name from a string
   function Make_Type_Name (S : String) return Type_Name
     with
       Pre  => S'Length <= Max_Type_Name_Length,
       Post => Make_Type_Name'Result.Length = S'Length;

   --  Compare type names
   function Equal (Left, Right : Type_Name) return Boolean;

   --  Lifetime representation (for Rust)
   type Lifetime_Name_Length is range 0 .. 32;
   type Lifetime is record
      Name      : String (1 .. 32) := (others => ' ');
      Name_Len  : Lifetime_Name_Length := 0;
      Is_Static : Boolean := False;
   end record;

   No_Lifetime : constant Lifetime := (Name => (others => ' '), Name_Len => 0, Is_Static => False);
   Static_Lifetime : constant Lifetime := (Name => "'static" & (8 .. 32 => ' '), Name_Len => 7, Is_Static => True);

   --  Forward declarations
   type STUNIR_Type;
   type Type_Access is access all STUNIR_Type;

   --  Struct field representation
   type Struct_Field is record
      Name       : Type_Name := Empty_Type_Name;
      Field_Kind : Type_Kind := Void_Kind;
      Visibility : Natural := 0;  --  0 = public, 1 = private, 2 = protected
      Offset     : Natural := 0;
   end record;

   --  Vector of struct fields
   package Field_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Struct_Field);

   subtype Field_Vector is Field_Vectors.Vector (Max_Fields);

   --  Enum variant representation
   type Enum_Variant is record
      Name  : Type_Name := Empty_Type_Name;
      Value : Integer := 0;
   end record;

   --  Vector of enum variants
   package Variant_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Enum_Variant);

   subtype Variant_Vector is Variant_Vectors.Vector (Max_Enum_Variants);

   --  Generic parameter vector
   package Generic_Vectors is new Ada.Containers.Formal_Vectors
     (Index_Type   => Positive,
      Element_Type => Type_Name);

   subtype Generic_Vector is Generic_Vectors.Vector (Max_Generic_Params);

   --  Core STUNIR Type discriminated record
   type STUNIR_Type (Kind : Type_Kind := Void_Kind) is record
      Name : Type_Name := Empty_Type_Name;
      case Kind is
         when Int_Kind =>
            Int_Width  : Int_Bits := 32;
            Is_Signed  : Boolean := True;

         when Float_Kind =>
            Float_Width : Float_Bits := 64;

         when Char_Kind =>
            Is_Unicode : Boolean := True;

         when String_Kind =>
            Is_Owned : Boolean := True;

         when Pointer_Kind =>
            Pointee_Kind : Type_Kind := Void_Kind;
            Ptr_Mutable  : Mutability_Kind := Mutable;
            Is_Nullable  : Boolean := True;

         when Reference_Kind =>
            Ref_Kind      : Type_Kind := Void_Kind;
            Ref_Mutable   : Mutability_Kind := Immutable;
            Ref_Lifetime  : Lifetime := No_Lifetime;

         when Array_Kind =>
            Element_Kind : Type_Kind := Void_Kind;
            Array_Size   : Natural := 0;  --  0 = dynamic

         when Slice_Kind =>
            Slice_Element : Type_Kind := Void_Kind;
            Slice_Mutable : Mutability_Kind := Immutable;
            Slice_Lifetime : Lifetime := No_Lifetime;

         when Struct_Kind =>
            Struct_Name   : Type_Name := Empty_Type_Name;
            Is_Packed     : Boolean := False;

         when Union_Kind =>
            Union_Name : Type_Name := Empty_Type_Name;

         when Enum_Kind =>
            Enum_Name : Type_Name := Empty_Type_Name;

         when Tagged_Union_Kind =>
            TU_Name : Type_Name := Empty_Type_Name;

         when Function_Kind =>
            Param_Count : Natural := 0;
            Is_Variadic : Boolean := False;
            Return_Kind : Type_Kind := Void_Kind;

         when Closure_Kind =>
            Closure_Params  : Natural := 0;
            Closure_Returns : Type_Kind := Void_Kind;

         when Generic_Kind =>
            Generic_Base : Type_Name := Empty_Type_Name;
            Arg_Count    : Natural := 0;

         when Type_Var_Kind =>
            Var_Name         : Type_Name := Empty_Type_Name;
            Constraint_Count : Natural := 0;

         when Opaque_Kind =>
            Opaque_Name : Type_Name := Empty_Type_Name;

         when Recursive_Kind =>
            Rec_Name : Type_Name := Empty_Type_Name;

         when Optional_Kind =>
            Inner_Kind : Type_Kind := Void_Kind;

         when Result_Kind =>
            Ok_Kind  : Type_Kind := Void_Kind;
            Err_Kind : Type_Kind := Void_Kind;

         when Tuple_Kind =>
            Tuple_Size : Natural := 0;

         when Void_Kind | Bool_Kind | Unit_Kind =>
            null;
      end case;
   end record;

   --  Default type values
   Void_Type : constant STUNIR_Type := (Kind => Void_Kind, Name => Empty_Type_Name);
   Unit_Type : constant STUNIR_Type := (Kind => Unit_Kind, Name => Empty_Type_Name);
   Bool_Type : constant STUNIR_Type := (Kind => Bool_Kind, Name => Empty_Type_Name);

   --  Common integer types
   function Make_Int_Type (Bits : Int_Bits; Signed : Boolean) return STUNIR_Type
     with
       Post => Make_Int_Type'Result.Kind = Int_Kind and
               Make_Int_Type'Result.Int_Width = Bits and
               Make_Int_Type'Result.Is_Signed = Signed;

   I8  : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 8, Is_Signed => True);
   I16 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 16, Is_Signed => True);
   I32 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 32, Is_Signed => True);
   I64 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 64, Is_Signed => True);
   U8  : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 8, Is_Signed => False);
   U16 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 16, Is_Signed => False);
   U32 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 32, Is_Signed => False);
   U64 : constant STUNIR_Type := (Kind => Int_Kind, Name => Empty_Type_Name, Int_Width => 64, Is_Signed => False);

   --  Common float types
   F32 : constant STUNIR_Type := (Kind => Float_Kind, Name => Empty_Type_Name, Float_Width => 32);
   F64 : constant STUNIR_Type := (Kind => Float_Kind, Name => Empty_Type_Name, Float_Width => 64);

   --  Type predicates
   function Is_Valid (T : STUNIR_Type) return Boolean
     with Post => Is_Valid'Result = True;  --  Always valid by construction

   function Is_Primitive (T : STUNIR_Type) return Boolean is
     (T.Kind in Void_Kind | Bool_Kind | Int_Kind | Float_Kind | Char_Kind | Unit_Kind);

   function Is_Pointer_Like (T : STUNIR_Type) return Boolean is
     (T.Kind in Pointer_Kind | Reference_Kind | Slice_Kind);

   function Is_Compound (T : STUNIR_Type) return Boolean is
     (T.Kind in Struct_Kind | Union_Kind | Enum_Kind | Tagged_Union_Kind | Tuple_Kind);

   function Is_Numeric (T : STUNIR_Type) return Boolean is
     (T.Kind in Int_Kind | Float_Kind);

   function Is_Integer (T : STUNIR_Type) return Boolean is
     (T.Kind = Int_Kind);

   function Is_Floating (T : STUNIR_Type) return Boolean is
     (T.Kind = Float_Kind);

   --  Type size calculations
   function Size_In_Bits (T : STUNIR_Type) return Natural
     with
       Pre => T.Kind in Int_Kind | Float_Kind | Bool_Kind | Char_Kind;

   --  Type equality
   function Types_Equal (Left, Right : STUNIR_Type) return Boolean;

   --  Type compatibility checking
   function Is_Compatible (Source, Target : STUNIR_Type) return Boolean;

   --  Type name operations
   function Get_Type_Name (T : STUNIR_Type) return String;
   function Get_IR_Type_String (T : STUNIR_Type) return String;

end Stunir_Types;
