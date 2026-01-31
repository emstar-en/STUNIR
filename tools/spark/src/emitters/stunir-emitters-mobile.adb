-- STUNIR Mobile App Emitter (SPARK Body)
with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Mobile is
   pragma SPARK_Mode (On);

   overriding procedure Emit_Module (Self : in out Mobile_Emitter; Module : in IR_Module; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      case Self.Config.Platform is
         when iOS_Swift =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Swift" & ASCII.LF & "import UIKit" & ASCII.LF & ASCII.LF);
         when Android_Kotlin =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Kotlin" & ASCII.LF & "package com.stunir" & ASCII.LF & ASCII.LF);
         when React_Native =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated React Native" & ASCII.LF & "import React from 'react';" & ASCII.LF);
         when Flutter =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Generated Flutter Dart" & ASCII.LF & "import 'package:flutter/material.dart';" & ASCII.LF);
         when others =>
            Code_Buffers.Append (Source => Output, New_Item => "// STUNIR Mobile" & ASCII.LF);
      end case;
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type (Self : in out Mobile_Emitter; T : in IR_Type_Def; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "struct " & Name_Strings.To_String (T.Name) & " {}" & ASCII.LF);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function (Self : in out Mobile_Emitter; Func : in IR_Function; Output : out IR_Code_Buffer; Success: out Boolean) is
   begin
      Code_Buffers.Set_Bounded_String (Target => Output, Source => "", Drop => Right);
      Code_Buffers.Append (Source => Output, New_Item => "func " & Name_Strings.To_String (Func.Name) & "() {}" & ASCII.LF);
      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Mobile;
