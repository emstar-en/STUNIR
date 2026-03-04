--  Type Map Runtime - Reusable type mapping for emitters
--  Maps STUNIR internal type names to target language type strings
--  Phase: 3 (Emit) - Library unit for emitter integration
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package Type_Map_Runtime is

   --  Map a STUNIR internal type name to target language type string
   --  Returns the mapped type string, or the original if no mapping exists
   function Map_Type
     (Type_Name : String;
      Target    : Target_Language) return String;

   --  Map using bounded string types (for SPARK compatibility)
   function Map_Type_Bounded
     (Type_Name : Type_Name_String;
      Target    : Target_Language) return Type_Name_String;

   --  Check if a type is a primitive (has standard mapping)
   function Is_Primitive_Type (Type_Name : String) return Boolean;

   --  Get default value for a type in target language
   --  Returns empty string if no default known
   function Get_Default_Value
     (Type_Name : String;
      Target    : Target_Language) return String;

private

   --  Internal primitive type enumeration for fast lookup
   type Primitive_Type_Id is
     (Prim_Void,
      Prim_Bool,
      Prim_I8, Prim_I16, Prim_I32, Prim_I64,
      Prim_U8, Prim_U16, Prim_U32, Prim_U64,
      Prim_F32, Prim_F64,
      Prim_Char, Prim_Str,
      Prim_Size,
      Prim_Ptr,
      Prim_Unknown);

   --  Classify a type name into primitive category
   function Classify_Type (Type_Name : String) return Primitive_Type_Id;

end Type_Map_Runtime;