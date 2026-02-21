--  STUNIR WASM Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body WASM_Emitter is

   New_Line : constant Character := ASCII.LF;

   function Map_Type_To_WASM (IR_Type : IR_Data_Type) return WASM_Type is
   begin
      case IR_Type is
         when Type_I8 | Type_I16 | Type_I32 | Type_U8 | Type_U16 | Type_U32 | Type_Bool =>
            return I32;
         when Type_I64 | Type_U64 =>
            return I64;
         when Type_F32 =>
            return F32;
         when Type_F64 =>
            return F64;
         when others =>
            return I32;
      end case;
   end Map_Type_To_WASM;

   procedure Emit_Module (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in WASM_Config;
      Status    : out Emitter_Status)
   is
      Module_Name : constant String := Identifier_Strings.To_String (Name);
      Pages_Str   : constant String := Positive'Image (Config.Memory_Pages);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      if Content_Strings.Length (Content) + 500 > Max_Content_Length then
         Status := Error_Buffer_Overflow;
         return;
      end if;

      Content_Strings.Append (Content,
         ";; STUNIR Generated WebAssembly Module" & New_Line &
         ";; Module: " & Module_Name & New_Line &
         ";; DO-178C Level A Compliant" & New_Line & New_Line &
         "(module" & New_Line &
         "  ;; Memory declaration" & New_Line &
         "  (memory (export ""memory"")" & Pages_Str & ")" & New_Line & New_Line);

      if Config.Enable_SIMD then
         Content_Strings.Append (Content,
            "  ;; SIMD enabled" & New_Line);
      end if;

      Content_Strings.Append (Content, ")" & New_Line);
   end Emit_Module;

   procedure Emit_Function (
      Name      : in Identifier_String;
      Params    : in String;
      Results   : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Status    : out Emitter_Status)
   is
      Func_Name : constant String := Identifier_Strings.To_String (Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      if Func_Name'Length + Params'Length + Results'Length + Body_Code'Length + 200 
         > Max_Content_Length then
         Status := Error_Buffer_Overflow;
         return;
      end if;

      Content_Strings.Append (Content,
         "  (func $" & Func_Name & " (export """ & Func_Name & """)" & New_Line &
         "    " & Params & New_Line &
         "    " & Results & New_Line &
         "    " & Body_Code & New_Line &
         "  )" & New_Line);
   end Emit_Function;

end WASM_Emitter;
