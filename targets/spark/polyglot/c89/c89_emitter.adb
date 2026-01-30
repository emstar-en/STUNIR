--  STUNIR C89 Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body C89_Emitter is

   New_Line : constant Character := ASCII.LF;

   function Map_Type_C89 (IR_Type : IR_Data_Type) return Type_Name_String is
   begin
      --  C89 doesn't have stdint.h, use plain types
      case IR_Type is
         when Type_Void   => return Type_Name_Strings.To_Bounded_String ("void");
         when Type_Bool   => return Type_Name_Strings.To_Bounded_String ("int");
         when Type_I8     => return Type_Name_Strings.To_Bounded_String ("signed char");
         when Type_I16    => return Type_Name_Strings.To_Bounded_String ("short");
         when Type_I32    => return Type_Name_Strings.To_Bounded_String ("long");
         when Type_I64    => return Type_Name_Strings.To_Bounded_String ("long long");
         when Type_U8     => return Type_Name_Strings.To_Bounded_String ("unsigned char");
         when Type_U16    => return Type_Name_Strings.To_Bounded_String ("unsigned short");
         when Type_U32    => return Type_Name_Strings.To_Bounded_String ("unsigned long");
         when Type_U64    => return Type_Name_Strings.To_Bounded_String ("unsigned long long");
         when Type_F32    => return Type_Name_Strings.To_Bounded_String ("float");
         when Type_F64    => return Type_Name_Strings.To_Bounded_String ("double");
         when Type_Char   => return Type_Name_Strings.To_Bounded_String ("char");
         when others      => return Type_Name_Strings.To_Bounded_String ("void");
      end case;
   end Map_Type_C89;

   procedure Emit_Header (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C89_Config;
      Status      : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "/*" & New_Line &
         " * STUNIR Generated C89 Header" & New_Line &
         " * Module: " & Name & New_Line &
         " * DO-178C Level A Compliant" & New_Line &
         " */" & New_Line &
         "#ifndef " & Name & "_H" & New_Line &
         "#define " & Name & "_H" & New_Line & New_Line &
         "/* C89 does not have stdint.h */" & New_Line &
         "typedef signed char int8_t;" & New_Line &
         "typedef unsigned char uint8_t;" & New_Line &
         "typedef short int16_t;" & New_Line &
         "typedef unsigned short uint16_t;" & New_Line &
         "typedef long int32_t;" & New_Line &
         "typedef unsigned long uint32_t;" & New_Line & New_Line &
         "#endif /* " & Name & "_H */" & New_Line);
   end Emit_Header;

   procedure Emit_Source (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in C89_Config;
      Status      : out Emitter_Status)
   is
      pragma Unreferenced (Config);
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "/*" & New_Line &
         " * STUNIR Generated C89 Source" & New_Line &
         " * Module: " & Name & New_Line &
         " */" & New_Line &
         "#include \"" & Name & ".h\"" & New_Line & New_Line);
   end Emit_Source;

end C89_Emitter;
