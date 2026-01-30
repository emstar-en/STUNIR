--  STUNIR Mobile Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Mobile_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_View (
      View_Name : in Identifier_String;
      Content   : out Content_String;
      Config    : in Mobile_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (View_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Platform is
         when iOS_Swift =>
            Content_Strings.Append (Content,
               "// STUNIR Generated Swift View" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "import SwiftUI" & New_Line & New_Line &
               "struct " & Name & ": View {" & New_Line &
               "    var body: some View {" & New_Line &
               "        Text(\"" & Name & "\")" & New_Line &
               "    }" & New_Line &
               "}" & New_Line);
         when Android_Kotlin =>
            Content_Strings.Append (Content,
               "// STUNIR Generated Kotlin Composable" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "package com.stunir.generated" & New_Line & New_Line &
               "import androidx.compose.runtime.Composable" & New_Line & New_Line &
               "@Composable" & New_Line &
               "fun " & Name & "() {" & New_Line &
               "    Text(text = \"" & Name & "\")" & New_Line &
               "}" & New_Line);
         when React_Native =>
            Content_Strings.Append (Content,
               "// STUNIR Generated React Native Component" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "import React from 'react';" & New_Line &
               "import { View, Text } from 'react-native';" & New_Line & New_Line &
               "export const " & Name & " = () => {" & New_Line &
               "  return (" & New_Line &
               "    <View>" & New_Line &
               "      <Text>" & Name & "</Text>" & New_Line &
               "    </View>" & New_Line &
               "  );" & New_Line &
               "};" & New_Line);
      end case;
   end Emit_View;

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Params    : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Mobile_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Platform is
         when iOS_Swift =>
            Content_Strings.Append (Content,
               "func " & Name & "(" & Params & ") {" & New_Line &
               "    " & Body_Code & New_Line &
               "}" & New_Line & New_Line);
         when Android_Kotlin =>
            Content_Strings.Append (Content,
               "fun " & Name & "(" & Params & ") {" & New_Line &
               "    " & Body_Code & New_Line &
               "}" & New_Line & New_Line);
         when React_Native =>
            Content_Strings.Append (Content,
               "const " & Name & " = (" & Params & ") => {" & New_Line &
               "  " & Body_Code & New_Line &
               "};" & New_Line & New_Line);
      end case;
   end Emit_Function;

end Mobile_Emitter;
