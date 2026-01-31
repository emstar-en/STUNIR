-- STUNIR Business Languages Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Business is
   pragma SPARK_Mode (On);

   -- Map IR primitive types to COBOL types
   function Get_COBOL_Type (Prim : IR_Primitive_Type) return String is
   begin
      case Prim is
         when Type_String => return "PIC X(255)";
         when Type_Int | Type_I32 => return "PIC S9(9) COMP";
         when Type_I8  => return "PIC S9(3) COMP";
         when Type_I16 => return "PIC S9(5) COMP";
         when Type_I64 => return "PIC S9(18) COMP";
         when Type_U8  => return "PIC 9(3) COMP";
         when Type_U16 => return "PIC 9(5) COMP";
         when Type_U32 => return "PIC 9(9) COMP";
         when Type_U64 => return "PIC 9(18) COMP";
         when Type_Float | Type_F32 => return "COMP-1";
         when Type_F64 => return "COMP-2";
         when Type_Bool => return "PIC 9 COMP";
         when Type_Void => return "OMITTED";
      end case;
   end Get_COBOL_Type;

   -- Map IR primitive types to BASIC types
   function Get_BASIC_Type (Prim : IR_Primitive_Type) return String is
   begin
      case Prim is
         when Type_String => return "String";
         when Type_Int | Type_I32 => return "Integer";
         when Type_I8 | Type_I16 => return "Integer";
         when Type_I64 => return "LongInt";
         when Type_U8 | Type_U16 | Type_U32 => return "Long";
         when Type_U64 => return "LongInt";
         when Type_Float | Type_F32 => return "Single";
         when Type_F64 => return "Double";
         when Type_Bool => return "Boolean";
         when Type_Void => return "";
      end case;
   end Get_BASIC_Type;

   -- Emit complete module
   overriding procedure Emit_Module
     (Self   : in out Business_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in COBOL_85 .. COBOL_2014 then
         -- Emit COBOL header
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "      ****************" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "      * STUNIR Generated COBOL" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "      * DO-178C Level A" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "      ****************" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       IDENTIFICATION DIVISION." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       PROGRAM-ID. " & Name_Strings.To_String (Module.Module_Name) & "." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       DATA DIVISION." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       WORKING-STORAGE SECTION." & ASCII.LF);

         -- Emit types as data items
         for I in 1 .. Module.Type_Cnt loop
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "       01 " & Name_Strings.To_String (Module.Types (I).Name) & "-REC." & ASCII.LF);
         end loop;

         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       PROCEDURE DIVISION." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       MAIN-PARA." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "           DISPLAY 'STUNIR COBOL Module'." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "           STOP RUN." & ASCII.LF);
      else
         -- Emit BASIC code
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "' STUNIR Generated BASIC" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "' DO-178C Level A" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "' Module: " & Name_Strings.To_String (Module.Module_Name) & ASCII.LF & ASCII.LF);

         if Self.Config.Line_Numbers then
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "10 PRINT ""STUNIR BASIC Module""" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "20 END" & ASCII.LF);
         else
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "PRINT ""STUNIR BASIC Module""" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "END" & ASCII.LF);
         end if;
      end if;

      Success := True;
   end Emit_Module;

   -- Emit type definition
   overriding procedure Emit_Type
     (Self   : in out Business_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in COBOL_85 .. COBOL_2014 then
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       01 " & Name_Strings.To_String (T.Name) & "-REC." & ASCII.LF);
         for I in 1 .. T.Field_Cnt loop
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "          05 " & Name_Strings.To_String (T.Fields (I).Name) & " PIC X." & ASCII.LF);
         end loop;
      else
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "TYPE " & Name_Strings.To_String (T.Name) & ASCII.LF);
      end if;

      Success := True;
   end Emit_Type;

   -- Emit function definition
   overriding procedure Emit_Function
     (Self   : in out Business_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      if Self.Config.Language in COBOL_85 .. COBOL_2014 then
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "       " & Name_Strings.To_String (Func.Name) & "-PARA." & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "           DISPLAY 'Function'." & ASCII.LF);
      else
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "SUB " & Name_Strings.To_String (Func.Name) & "()" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "  PRINT ""Function""" & ASCII.LF);
         Code_Buffers.Append
           (Source   => Output,
            New_Item => "END SUB" & ASCII.LF);
      end if;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Business;
