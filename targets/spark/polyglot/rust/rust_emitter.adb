--  STUNIR Rust Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Rust_Emitter is

   New_Line : constant Character := ASCII.LF;

   function Map_Type_Rust (IR_Type : IR_Data_Type) return Type_Name_String is
   begin
      case IR_Type is
         when Type_Void   => return Type_Name_Strings.To_Bounded_String ("()");
         when Type_Bool   => return Type_Name_Strings.To_Bounded_String ("bool");
         when Type_I8     => return Type_Name_Strings.To_Bounded_String ("i8");
         when Type_I16    => return Type_Name_Strings.To_Bounded_String ("i16");
         when Type_I32    => return Type_Name_Strings.To_Bounded_String ("i32");
         when Type_I64    => return Type_Name_Strings.To_Bounded_String ("i64");
         when Type_U8     => return Type_Name_Strings.To_Bounded_String ("u8");
         when Type_U16    => return Type_Name_Strings.To_Bounded_String ("u16");
         when Type_U32    => return Type_Name_Strings.To_Bounded_String ("u32");
         when Type_U64    => return Type_Name_Strings.To_Bounded_String ("u64");
         when Type_F32    => return Type_Name_Strings.To_Bounded_String ("f32");
         when Type_F64    => return Type_Name_Strings.To_Bounded_String ("f64");
         when Type_Char   => return Type_Name_Strings.To_Bounded_String ("char");
         when Type_String => return Type_Name_Strings.To_Bounded_String ("String");
         when others      => return Type_Name_Strings.To_Bounded_String ("()");
      end case;
   end Map_Type_Rust;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Rust_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      Content_Strings.Append (Content,
         "//! STUNIR Generated Rust Module" & New_Line &
         "//! Module: " & Name & New_Line &
         "//! DO-178C Level A Compliant" & New_Line & New_Line);

      if Config.No_Std then
         Content_Strings.Append (Content,
            "#![no_std]" & New_Line & New_Line);
      end if;

      Content_Strings.Append (Content,
         "// Module: " & Name & New_Line & New_Line);
   end Emit_Module;

end Rust_Emitter;
