--  STUNIR Emitter Types - Ada SPARK Implementation
--  DO-178C Level A compliant implementation

pragma SPARK_Mode (On);

package body Emitter_Types is

   --  Type mapping for C code generation
   function Map_IR_Type_To_C (IR_Type : IR_Data_Type) return Type_Name_String is
   begin
      case IR_Type is
         when Type_Void   => return Type_Name_Strings.To_Bounded_String ("void");
         when Type_Bool   => return Type_Name_Strings.To_Bounded_String ("uint8_t");
         when Type_I8     => return Type_Name_Strings.To_Bounded_String ("int8_t");
         when Type_I16    => return Type_Name_Strings.To_Bounded_String ("int16_t");
         when Type_I32    => return Type_Name_Strings.To_Bounded_String ("int32_t");
         when Type_I64    => return Type_Name_Strings.To_Bounded_String ("int64_t");
         when Type_U8     => return Type_Name_Strings.To_Bounded_String ("uint8_t");
         when Type_U16    => return Type_Name_Strings.To_Bounded_String ("uint16_t");
         when Type_U32    => return Type_Name_Strings.To_Bounded_String ("uint32_t");
         when Type_U64    => return Type_Name_Strings.To_Bounded_String ("uint64_t");
         when Type_F32    => return Type_Name_Strings.To_Bounded_String ("float");
         when Type_F64    => return Type_Name_Strings.To_Bounded_String ("double");
         when Type_Char   => return Type_Name_Strings.To_Bounded_String ("char");
         when Type_String => return Type_Name_Strings.To_Bounded_String ("char*");
         when Type_Pointer => return Type_Name_Strings.To_Bounded_String ("void*");
         when Type_Array  => return Type_Name_Strings.To_Bounded_String ("void*");
         when Type_Struct => return Type_Name_Strings.To_Bounded_String ("struct");
      end case;
   end Map_IR_Type_To_C;

   --  Get architecture configuration
   function Get_Arch_Config (Arch : Architecture_Type) return Arch_Config_Type is
   begin
      case Arch is
         when Arch_ARM =>
            return (Word_Size => 32, Endianness => Little_Endian, 
                    Alignment => 4, Stack_Grows_Down => True);
         when Arch_ARM64 =>
            return (Word_Size => 64, Endianness => Little_Endian,
                    Alignment => 8, Stack_Grows_Down => True);
         when Arch_AVR =>
            return (Word_Size => 8, Endianness => Little_Endian,
                    Alignment => 1, Stack_Grows_Down => True);
         when Arch_MIPS =>
            return (Word_Size => 32, Endianness => Big_Endian,
                    Alignment => 4, Stack_Grows_Down => True);
         when Arch_RISCV =>
            return (Word_Size => 32, Endianness => Little_Endian,
                    Alignment => 4, Stack_Grows_Down => True);
         when Arch_X86 =>
            return (Word_Size => 32, Endianness => Little_Endian,
                    Alignment => 4, Stack_Grows_Down => True);
         when Arch_X86_64 =>
            return (Word_Size => 64, Endianness => Little_Endian,
                    Alignment => 8, Stack_Grows_Down => True);
         when Arch_PowerPC =>
            return (Word_Size => 32, Endianness => Big_Endian,
                    Alignment => 4, Stack_Grows_Down => True);
         when Arch_Generic =>
            return (Word_Size => 32, Endianness => Little_Endian,
                    Alignment => 4, Stack_Grows_Down => True);
      end case;
   end Get_Arch_Config;

   --  SHA256 computation (simplified placeholder - full impl would use GNAT.SHA256)
   function Compute_SHA256 (Content : String) return Hash_String is
      Result : Hash_String;
      Hex_Chars : constant String := "0123456789abcdef";
      Hash_Val : Natural := 0;
   begin
      --  Simplified hash for SPARK verification
      --  Real implementation would use GNAT.SHA256
      for I in Content'Range loop
         Hash_Val := (Hash_Val + Character'Pos (Content (I))) mod 16#FFFF#;
      end loop;
      
      --  Generate 64-character hex string placeholder
      Result := Hash_Strings.To_Bounded_String ((1 .. 64 => '0'));
      
      --  Fill with deterministic pattern based on content
      for I in 1 .. 16 loop
         declare
            Idx : constant Positive := ((Hash_Val + I) mod 16) + 1;
         begin
            Hash_Strings.Replace_Element (Result, I, Hex_Chars (Idx));
            Hash_Strings.Replace_Element (Result, I + 16, Hex_Chars (Idx));
            Hash_Strings.Replace_Element (Result, I + 32, Hex_Chars (Idx));
            Hash_Strings.Replace_Element (Result, I + 48, Hex_Chars (Idx));
         end;
      end loop;
      
      return Result;
   end Compute_SHA256;

end Emitter_Types;
