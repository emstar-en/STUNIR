--  Type Map Runtime - Reusable type mapping for emitters
--  Maps STUNIR internal type names to target language type strings
--  Phase: 3 (Emit) - Library unit for emitter integration
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Ada.Characters.Handling;
with STUNIR_Types;
use STUNIR_Types;

package body Type_Map_Runtime is

   use Ada.Characters.Handling;

   --  Classify a type name into primitive category
   function Classify_Type (Type_Name : String) return Primitive_Type_Id is
      Lower : constant String := To_Lower (Type_Name);
   begin
      if Lower = "void" then
         return Prim_Void;
      elsif Lower = "bool" or Lower = "boolean" or Lower = "_bool" then
         return Prim_Bool;
      elsif Lower = "i8" or Lower = "int8" or Lower = "int8_t" then
         return Prim_I8;
      elsif Lower = "i16" or Lower = "int16" or Lower = "int16_t" then
         return Prim_I16;
      elsif Lower = "i32" or Lower = "int32" or Lower = "int32_t" or Lower = "int" then
         return Prim_I32;
      elsif Lower = "i64" or Lower = "int64" or Lower = "int64_t" or Lower = "long" then
         return Prim_I64;
      elsif Lower = "u8" or Lower = "uint8" or Lower = "uint8_t" or Lower = "byte" then
         return Prim_U8;
      elsif Lower = "u16" or Lower = "uint16" or Lower = "uint16_t" then
         return Prim_U16;
      elsif Lower = "u32" or Lower = "uint32" or Lower = "uint32_t" then
         return Prim_U32;
      elsif Lower = "u64" or Lower = "uint64" or Lower = "uint64_t" then
         return Prim_U64;
      elsif Lower = "f32" or Lower = "float" or Lower = "float32" then
         return Prim_F32;
      elsif Lower = "f64" or Lower = "double" or Lower = "float64" then
         return Prim_F64;
      elsif Lower = "char" then
         return Prim_Char;
      elsif Lower = "str" or Lower = "string" then
         return Prim_Str;
      elsif Lower = "size_t" or Lower = "usize" then
         return Prim_Size;
      elsif Lower = "ptr" or Lower = "pointer" or Lower = "void*" then
         return Prim_Ptr;
      else
         return Prim_Unknown;
      end if;
   end Classify_Type;

   --  Map primitive type to target language string
   function Map_Primitive
     (Prim    : Primitive_Type_Id;
      Target  : Target_Language) return String is
   begin
      case Target is
         --  C
         when Target_C =>
            case Prim is
               when Prim_Void    => return "void";
               when Prim_Bool    => return "_Bool";
               when Prim_I8      => return "int8_t";
               when Prim_I16     => return "int16_t";
               when Prim_I32     => return "int32_t";
               when Prim_I64     => return "int64_t";
               when Prim_U8      => return "uint8_t";
               when Prim_U16     => return "uint16_t";
               when Prim_U32     => return "uint32_t";
               when Prim_U64     => return "uint64_t";
               when Prim_F32     => return "float";
               when Prim_F64     => return "double";
               when Prim_Char    => return "char";
               when Prim_Str    => return "const char*";
               when Prim_Size    => return "size_t";
               when Prim_Ptr     => return "void*";
               when Prim_Unknown => return "";
            end case;

         --  C++
         when Target_CPP =>
            case Prim is
               when Prim_Void    => return "void";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "int8_t";
               when Prim_I16     => return "int16_t";
               when Prim_I32     => return "int32_t";
               when Prim_I64     => return "int64_t";
               when Prim_U8      => return "uint8_t";
               when Prim_U16     => return "uint16_t";
               when Prim_U32     => return "uint32_t";
               when Prim_U64     => return "uint64_t";
               when Prim_F32     => return "float";
               when Prim_F64     => return "double";
               when Prim_Char    => return "char";
               when Prim_Str    => return "std::string";
               when Prim_Size    => return "size_t";
               when Prim_Ptr     => return "void*";
               when Prim_Unknown => return "";
            end case;

         --  Python
         when Target_Python =>
            case Prim is
               when Prim_Void    => return "None";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "int";
               when Prim_I16     => return "int";
               when Prim_I32     => return "int";
               when Prim_I64     => return "int";
               when Prim_U8      => return "int";
               when Prim_U16     => return "int";
               when Prim_U32     => return "int";
               when Prim_U64     => return "int";
               when Prim_F32     => return "float";
               when Prim_F64     => return "float";
               when Prim_Char    => return "str";
               when Prim_Str    => return "str";
               when Prim_Size    => return "int";
               when Prim_Ptr     => return "ctypes.c_void_p";
               when Prim_Unknown => return "";
            end case;

         --  Rust
         when Target_Rust =>
            case Prim is
               when Prim_Void    => return "()";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "i8";
               when Prim_I16     => return "i16";
               when Prim_I32     => return "i32";
               when Prim_I64     => return "i64";
               when Prim_U8      => return "u8";
               when Prim_U16     => return "u16";
               when Prim_U32     => return "u32";
               when Prim_U64     => return "u64";
               when Prim_F32     => return "f32";
               when Prim_F64     => return "f64";
               when Prim_Char    => return "char";
               when Prim_Str    => return "&str";
               when Prim_Size    => return "usize";
               when Prim_Ptr     => return "*const c_void";
               when Prim_Unknown => return "";
            end case;

         --  Go
         when Target_Go =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "int8";
               when Prim_I16     => return "int16";
               when Prim_I32     => return "int32";
               when Prim_I64     => return "int64";
               when Prim_U8      => return "uint8";
               when Prim_U16     => return "uint16";
               when Prim_U32     => return "uint32";
               when Prim_U64     => return "uint64";
               when Prim_F32     => return "float32";
               when Prim_F64     => return "float64";
               when Prim_Char    => return "rune";
               when Prim_Str    => return "string";
               when Prim_Size    => return "uintptr";
               when Prim_Ptr     => return "unsafe.Pointer";
               when Prim_Unknown => return "";
            end case;

         --  Java
         when Target_Java =>
            case Prim is
               when Prim_Void    => return "void";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "byte";
               when Prim_I16     => return "short";
               when Prim_I32     => return "int";
               when Prim_I64     => return "long";
               when Prim_U8      => return "int";  --  Java has no unsigned
               when Prim_U16     => return "int";
               when Prim_U32     => return "long";
               when Prim_U64     => return "long";
               when Prim_F32     => return "float";
               when Prim_F64     => return "double";
               when Prim_Char    => return "char";
               when Prim_Str    => return "String";
               when Prim_Size    => return "long";
               when Prim_Ptr     => return "Object";
               when Prim_Unknown => return "";
            end case;

         --  JavaScript
         when Target_JavaScript =>
            case Prim is
               when Prim_Void    => return "undefined";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "number";
               when Prim_I16     => return "number";
               when Prim_I32     => return "number";
               when Prim_I64     => return "bigint";
               when Prim_U8      => return "number";
               when Prim_U16     => return "number";
               when Prim_U32     => return "number";
               when Prim_U64     => return "bigint";
               when Prim_F32     => return "number";
               when Prim_F64     => return "number";
               when Prim_Char    => return "string";
               when Prim_Str    => return "string";
               when Prim_Size    => return "number";
               when Prim_Ptr     => return "object";
               when Prim_Unknown => return "";
            end case;

         --  C#
         when Target_CSharp =>
            case Prim is
               when Prim_Void    => return "void";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "sbyte";
               when Prim_I16     => return "short";
               when Prim_I32     => return "int";
               when Prim_I64     => return "long";
               when Prim_U8      => return "byte";
               when Prim_U16     => return "ushort";
               when Prim_U32     => return "uint";
               when Prim_U64     => return "ulong";
               when Prim_F32     => return "float";
               when Prim_F64     => return "double";
               when Prim_Char    => return "char";
               when Prim_Str    => return "string";
               when Prim_Size    => return "nuint";
               when Prim_Ptr     => return "IntPtr";
               when Prim_Unknown => return "";
            end case;

         --  Swift
         when Target_Swift =>
            case Prim is
               when Prim_Void    => return "Void";
               when Prim_Bool    => return "Bool";
               when Prim_I8      => return "Int8";
               when Prim_I16     => return "Int16";
               when Prim_I32     => return "Int32";
               when Prim_I64     => return "Int64";
               when Prim_U8      => return "UInt8";
               when Prim_U16     => return "UInt16";
               when Prim_U32     => return "UInt32";
               when Prim_U64     => return "UInt64";
               when Prim_F32     => return "Float";
               when Prim_F64     => return "Double";
               when Prim_Char    => return "Character";
               when Prim_Str    => return "String";
               when Prim_Size    => return "Int";
               when Prim_Ptr     => return "UnsafeRawPointer";
               when Prim_Unknown => return "";
            end case;

         --  Kotlin
         when Target_Kotlin =>
            case Prim is
               when Prim_Void    => return "Unit";
               when Prim_Bool    => return "Boolean";
               when Prim_I8      => return "Byte";
               when Prim_I16     => return "Short";
               when Prim_I32     => return "Int";
               when Prim_I64     => return "Long";
               when Prim_U8      => return "UInt";  --  Kotlin unsigned
               when Prim_U16     => return "UShort";
               when Prim_U32     => return "UInt";
               when Prim_U64     => return "ULong";
               when Prim_F32     => return "Float";
               when Prim_F64     => return "Double";
               when Prim_Char    => return "Char";
               when Prim_Str    => return "String";
               when Prim_Size    => return "ULong";
               when Prim_Ptr     => return "Any?";
               when Prim_Unknown => return "";
            end case;

         --  SPARK/Ada
         when Target_SPARK | Target_Ada =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "Boolean";
               when Prim_I8      => return "Interfaces.Integer_8";
               when Prim_I16     => return "Interfaces.Integer_16";
               when Prim_I32     => return "Interfaces.Integer_32";
               when Prim_I64     => return "Interfaces.Integer_64";
               when Prim_U8      => return "Interfaces.Unsigned_8";
               when Prim_U16     => return "Interfaces.Unsigned_16";
               when Prim_U32     => return "Interfaces.Unsigned_32";
               when Prim_U64     => return "Interfaces.Unsigned_64";
               when Prim_F32     => return "Interfaces.IEEE_Float_32";
               when Prim_F64     => return "Interfaces.IEEE_Float_64";
               when Prim_Char    => return "Character";
               when Prim_Str    => return "String";
               when Prim_Size    => return "Natural";
               when Prim_Ptr     => return "System.Address";
               when Prim_Unknown => return "";
            end case;

         --  Common Lisp
         when Target_Common_Lisp =>
            case Prim is
               when Prim_Void    => return "null";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "(signed-byte 8)";
               when Prim_I16     => return "(signed-byte 16)";
               when Prim_I32     => return "(signed-byte 32)";
               when Prim_I64     => return "(signed-byte 64)";
               when Prim_U8      => return "(unsigned-byte 8)";
               when Prim_U16     => return "(unsigned-byte 16)";
               when Prim_U32     => return "(unsigned-byte 32)";
               when Prim_U64     => return "(unsigned-byte 64)";
               when Prim_F32     => return "single-float";
               when Prim_F64     => return "double-float";
               when Prim_Char    => return "character";
               when Prim_Str    => return "string";
               when Prim_Size    => return "fixnum";
               when Prim_Ptr     => return "sb-sys:system-area-pointer";
               when Prim_Unknown => return "t";
            end case;

         --  Scheme / Racket / Guile
         when Target_Scheme | Target_Racket | Target_Guile =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "integer";
               when Prim_I16     => return "integer";
               when Prim_I32     => return "integer";
               when Prim_I64     => return "integer";
               when Prim_U8      => return "integer";
               when Prim_U16     => return "integer";
               when Prim_U32     => return "integer";
               when Prim_U64     => return "integer";
               when Prim_F32     => return "real";
               when Prim_F64     => return "real";
               when Prim_Char    => return "char";
               when Prim_Str    => return "string";
               when Prim_Size    => return "integer";
               when Prim_Ptr     => return "";
               when Prim_Unknown => return "";
            end case;

         --  Emacs Lisp
         when Target_Emacs_Lisp =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "integer";
               when Prim_I16     => return "integer";
               when Prim_I32     => return "integer";
               when Prim_I64     => return "integer";
               when Prim_U8      => return "integer";
               when Prim_U16     => return "integer";
               when Prim_U32     => return "integer";
               when Prim_U64     => return "integer";
               when Prim_F32     => return "float";
               when Prim_F64     => return "float";
               when Prim_Char    => return "char";
               when Prim_Str    => return "string";
               when Prim_Size    => return "integer";
               when Prim_Ptr     => return "";
               when Prim_Unknown => return "";
            end case;

         --  Clojure
         when Target_Clojure =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "Boolean";
               when Prim_I8      => return "Byte";
               when Prim_I16     => return "Short";
               when Prim_I32     => return "Integer";
               when Prim_I64     => return "Long";
               when Prim_U8      => return "Integer";
               when Prim_U16     => return "Integer";
               when Prim_U32     => return "Long";
               when Prim_U64     => return "Long";
               when Prim_F32     => return "Float";
               when Prim_F64     => return "Double";
               when Prim_Char    => return "Character";
               when Prim_Str    => return "String";
               when Prim_Size    => return "Long";
               when Prim_Ptr     => return "Object";
               when Prim_Unknown => return "Object";
            end case;

         --  SWI-Prolog / GNU Prolog
         when Target_SWI_Prolog | Target_GNU_Prolog =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "integer";
               when Prim_I16     => return "integer";
               when Prim_I32     => return "integer";
               when Prim_I64     => return "integer";
               when Prim_U8      => return "integer";
               when Prim_U16     => return "integer";
               when Prim_U32     => return "integer";
               when Prim_U64     => return "integer";
               when Prim_F32     => return "float";
               when Prim_F64     => return "float";
               when Prim_Char    => return "atom";
               when Prim_Str    => return "atom";
               when Prim_Size    => return "integer";
               when Prim_Ptr     => return "";
               when Prim_Unknown => return "";
            end case;

         --  Mercury
         when Target_Mercury =>
            case Prim is
               when Prim_Void    => return "io";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "int";
               when Prim_I16     => return "int";
               when Prim_I32     => return "int";
               when Prim_I64     => return "int";
               when Prim_U8      => return "int";
               when Prim_U16     => return "int";
               when Prim_U32     => return "int";
               when Prim_U64     => return "int";
               when Prim_F32     => return "float";
               when Prim_F64     => return "float";
               when Prim_Char    => return "char";
               when Prim_Str    => return "string";
               when Prim_Size    => return "int";
               when Prim_Ptr     => return "c_pointer";
               when Prim_Unknown => return "";
            end case;

         --  Futhark
         when Target_Futhark =>
            case Prim is
               when Prim_Void    => return "unit";
               when Prim_Bool    => return "bool";
               when Prim_I8      => return "i8";
               when Prim_I16     => return "i16";
               when Prim_I32     => return "i32";
               when Prim_I64     => return "i64";
               when Prim_U8      => return "u8";
               when Prim_U16     => return "u16";
               when Prim_U32     => return "u32";
               when Prim_U64     => return "u64";
               when Prim_F32     => return "f32";
               when Prim_F64     => return "f64";
               when Prim_Char    => return "i8";  --  Futhark has no char type
               when Prim_Str    => return "[]u8";  --  Futhark uses byte arrays
               when Prim_Size    => return "i64";
               when Prim_Ptr     => return "";
               when Prim_Unknown => return "";
            end case;

         --  Lean4
         when Target_Lean4 =>
            case Prim is
               when Prim_Void    => return "Unit";
               when Prim_Bool    => return "Bool";
               when Prim_I8      => return "Int8";
               when Prim_I16     => return "Int16";
               when Prim_I32     => return "Int32";
               when Prim_I64     => return "Int64";
               when Prim_U8      => return "UInt8";
               when Prim_U16     => return "UInt16";
               when Prim_U32     => return "UInt32";
               when Prim_U64     => return "UInt64";
               when Prim_F32     => return "Float";
               when Prim_F64     => return "Float";
               when Prim_Char    => return "Char";
               when Prim_Str    => return "String";
               when Prim_Size    => return "Nat";
               when Prim_Ptr     => return "Unit";
               when Prim_Unknown => return "";
            end case;

         --  Haskell
         when Target_Haskell =>
            case Prim is
               when Prim_Void    => return "()";
               when Prim_Bool    => return "Bool";
               when Prim_I8      => return "Int8";
               when Prim_I16     => return "Int16";
               when Prim_I32     => return "Int32";
               when Prim_I64     => return "Int64";
               when Prim_U8      => return "Word8";
               when Prim_U16     => return "Word16";
               when Prim_U32     => return "Word32";
               when Prim_U64     => return "Word64";
               when Prim_F32     => return "Float";
               when Prim_F64     => return "Double";
               when Prim_Char    => return "Char";
               when Prim_Str    => return "String";
               when Prim_Size    => return "Int";
               when Prim_Ptr     => return "Ptr ()";
               when Prim_Unknown => return "";
            end case;

         --  Hy (Lisp on Python)
         when Target_Hy =>
            case Prim is
               when Prim_Void    => return "None";
               when Prim_Bool    => return "bool";
               when others       => return "int";  --  Hy uses Python types
            end case;

         --  Janet
         when Target_Janet =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "int";
               when Prim_I16     => return "int";
               when Prim_I32     => return "int";
               when Prim_I64     => return "int";
               when Prim_U8      => return "int";
               when Prim_U16     => return "int";
               when Prim_U32     => return "int";
               when Prim_U64     => return "int";
               when Prim_F32     => return "number";
               when Prim_F64     => return "number";
               when Prim_Char    => return "string";
               when Prim_Str    => return "string";
               when Prim_Size    => return "int";
               when Prim_Ptr     => return "";
               when Prim_Unknown => return "";
            end case;

         --  ClojureScript
         when Target_ClojureScript =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "boolean";
               when Prim_I8      => return "number";
               when Prim_I16     => return "number";
               when Prim_I32     => return "number";
               when Prim_I64     => return "number";
               when Prim_U8      => return "number";
               when Prim_U16     => return "number";
               when Prim_U32     => return "number";
               when Prim_U64     => return "number";
               when Prim_F32     => return "number";
               when Prim_F64     => return "number";
               when Prim_Char    => return "string";
               when Prim_Str    => return "string";
               when Prim_Size    => return "number";
               when Prim_Ptr     => return "object";
               when Prim_Unknown => return "";
            end case;

         --  Generic Prolog (deprecated)
         when Target_Prolog =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "boolean";
               when others       => return "integer";
            end case;
      end case;
   end Map_Primitive;

   --  Map a STUNIR internal type name to target language type string
   function Map_Type
     (Type_Name : String;
      Target    : Target_Language) return String
   is
      Prim : constant Primitive_Type_Id := Classify_Type (Type_Name);
      Mapped : constant String := Map_Primitive (Prim, Target);
   begin
      if Prim /= Prim_Unknown then
         return Mapped;
      else
         --  Custom type: return as-is
         return Type_Name;
      end if;
   end Map_Type;

   --  Map using bounded string types
   function Map_Type_Bounded
     (Type_Name : Type_Name_String;
      Target    : Target_Language) return Type_Name_String
   is
      Name_Str : constant String := Type_Name_Strings.To_String (Type_Name);
      Mapped   : constant String := Map_Type (Name_Str, Target);
   begin
      return Type_Name_Strings.To_Bounded_String (Mapped);
   end Map_Type_Bounded;

   --  Check if a type is a primitive
   function Is_Primitive_Type (Type_Name : String) return Boolean is
      Prim : constant Primitive_Type_Id := Classify_Type (Type_Name);
   begin
      return Prim /= Prim_Unknown;
   end Is_Primitive_Type;

   --  Get default value for a type in target language
   function Get_Default_Value
     (Type_Name : String;
      Target    : Target_Language) return String
   is
      Prim : constant Primitive_Type_Id := Classify_Type (Type_Name);
   begin
      if Prim = Prim_Unknown then
         return "";
      end if;

      case Target is
         when Target_C | Target_CPP =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "false";
               when Prim_I8      => return "0";
               when Prim_I16     => return "0";
               when Prim_I32     => return "0";
               when Prim_I64     => return "0";
               when Prim_U8      => return "0";
               when Prim_U16     => return "0";
               when Prim_U32     => return "0";
               when Prim_U64     => return "0";
               when Prim_F32     => return "0.0f";
               when Prim_F64     => return "0.0";
               when Prim_Char    => return "Character'Val(0)";
               when Prim_Str    => return "";
               when Prim_Size    => return "0";
               when Prim_Ptr     => return "NULL";
               when Prim_Unknown => return "";
            end case;

         when Target_Python =>
            case Prim is
               when Prim_Void    => return "None";
               when Prim_Bool    => return "False";
               when others       => return "0";
            end case;

         when Target_Rust =>
            case Prim is
               when Prim_Void    => return "()";
               when Prim_Bool    => return "false";
               when others       => return "0";
            end case;

         when Target_Futhark =>
            case Prim is
               when Prim_Void    => return "()";
               when Prim_Bool    => return "false";
               when others       => return "0";
            end case;

         when Target_Lean4 =>
            case Prim is
               when Prim_Void    => return "()";
               when Prim_Bool    => return "false";
               when others       => return "0";
            end case;

         when Target_Common_Lisp | Target_Scheme | Target_Racket | Target_Guile | Target_Emacs_Lisp =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "nil";
               when others       => return "0";
            end case;

         when Target_Clojure =>
            case Prim is
               when Prim_Void    => return "nil";
               when Prim_Bool    => return "false";
               when others       => return "0";
            end case;

         when Target_SWI_Prolog | Target_GNU_Prolog | Target_Mercury =>
            case Prim is
               when Prim_Void    => return "";
               when Prim_Bool    => return "false";
               when others       => return "0";
            end case;

         when others =>
            return "";
      end case;
   end Get_Default_Value;

end Type_Map_Runtime;