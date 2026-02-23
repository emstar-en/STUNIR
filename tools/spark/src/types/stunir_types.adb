--  STUNIR Types Package Body
--  DO-333 / SPARK 2014 compliant
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body STUNIR_Types is

   function Status_Code_Image (Status : Status_Code) return String is
   begin
      case Status is
         when Success                 => return "Success";
         when Error_File_Not_Found    => return "Error: File not found";
         when Error_File_Read         => return "Error: File read failed";
         when Error_File_Write        => return "Error: File write failed";
         when Error_Invalid_JSON      => return "Error: Invalid JSON";
         when Error_Invalid_Schema    => return "Error: Invalid schema";
         when Error_Invalid_Syntax    => return "Error: Invalid syntax";
         when Error_Buffer_Overflow   => return "Error: Buffer overflow";
         when Error_Unsupported_Type  => return "Error: Unsupported type";
         when Error_Parse_Error       => return "Error: Parse error";
         when Error_Validation_Failed => return "Error: Validation failed";
         when Error_Conversion_Failed => return "Error: Conversion failed";
         when Error_Emission_Failed   => return "Error: Code emission failed";
         when Error_Not_Implemented   => return "Error: Not implemented";
         when Error_Invalid_Format    => return "Error: Invalid format";
         when Error_Empty_Extraction  => return "Error: Empty extraction";
         when Error_Too_Large         => return "Error: Too large";
         when Error_Parse             => return "Error: Parse";
         when Error_File_IO           => return "Error: File I/O failed";
         when Error_Invalid_Input     => return "Error: Invalid input";
      end case;
   end Status_Code_Image;

   function Is_Success (Status : Status_Code) return Boolean is
   begin
      return Status = Success;
   end Is_Success;

   function Is_Error (Status : Status_Code) return Boolean is
   begin
      return Status /= Success;
   end Is_Error;

   function Make_Default_Step return IR_Step is
      Default_Case : constant Case_Entry := (
         Case_Value => Identifier_Strings.Null_Bounded_String,
         Body_Start => 0,
         Body_Count => 0
      );
      Default_Catch : constant Catch_Entry := (
         Exception_Type => Identifier_Strings.Null_Bounded_String,
         Var_Name       => Identifier_Strings.Null_Bounded_String,
         Body_Start     => 0,
         Body_Count     => 0
      );
   begin
      return (
         Step_Type      => Step_Nop,
         Target         => Identifier_Strings.Null_Bounded_String,
         Value          => Identifier_Strings.Null_Bounded_String,
         Condition      => Identifier_Strings.Null_Bounded_String,
         Init           => Identifier_Strings.Null_Bounded_String,
         Increment      => Identifier_Strings.Null_Bounded_String,
         Then_Start     => 0,
         Then_Count     => 0,
         Else_Start     => 0,
         Else_Count     => 0,
         Body_Start     => 0,
         Body_Count     => 0,
         Expr           => Identifier_Strings.Null_Bounded_String,
         Cases          => (others => Default_Case),
         Case_Count     => 0,
         Default_Start  => 0,
         Default_Count  => 0,
         Try_Start      => 0,
         Try_Count      => 0,
         Catch_Blocks   => (others => Default_Catch),
         Catch_Count    => 0,
         Finally_Start  => 0,
         Finally_Count  => 0,
         Index          => Identifier_Strings.Null_Bounded_String,
         Key            => Identifier_Strings.Null_Bounded_String,
         Field          => Identifier_Strings.Null_Bounded_String,
         Args           => Identifier_Strings.Null_Bounded_String,
         Type_Args      => Identifier_Strings.Null_Bounded_String,
         Error_Msg      => Identifier_Strings.Null_Bounded_String
      );
   end Make_Default_Step;

   procedure Init_Step_Collection (Steps : out Step_Collection) is
   begin
      Steps.Count := 0;
      for I in Step_Index range 1 .. Max_Steps loop
         --  Initialize each field individually to avoid stack overflow
         Steps.Steps (I).Step_Type := Step_Nop;
         Steps.Steps (I).Target := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Value := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Condition := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Init := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Increment := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Then_Start := 0;
         Steps.Steps (I).Then_Count := 0;
         Steps.Steps (I).Else_Start := 0;
         Steps.Steps (I).Else_Count := 0;
         Steps.Steps (I).Body_Start := 0;
         Steps.Steps (I).Body_Count := 0;
         Steps.Steps (I).Expr := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Case_Count := 0;
         Steps.Steps (I).Default_Start := 0;
         Steps.Steps (I).Default_Count := 0;
         Steps.Steps (I).Try_Start := 0;
         Steps.Steps (I).Try_Count := 0;
         Steps.Steps (I).Catch_Count := 0;
         Steps.Steps (I).Finally_Start := 0;
         Steps.Steps (I).Finally_Count := 0;
         Steps.Steps (I).Index := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Key := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Field := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Args := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Type_Args := Identifier_Strings.Null_Bounded_String;
         Steps.Steps (I).Error_Msg := Identifier_Strings.Null_Bounded_String;
         --  Initialize Cases array
         for C in 1 .. Max_Cases loop
            Steps.Steps (I).Cases (C).Case_Value := Identifier_Strings.Null_Bounded_String;
            Steps.Steps (I).Cases (C).Body_Start := 0;
            Steps.Steps (I).Cases (C).Body_Count := 0;
         end loop;
         --  Initialize Catch_Blocks array
         for C in 1 .. Max_Catch_Blocks loop
            Steps.Steps (I).Catch_Blocks (C).Exception_Type := Identifier_Strings.Null_Bounded_String;
            Steps.Steps (I).Catch_Blocks (C).Var_Name := Identifier_Strings.Null_Bounded_String;
            Steps.Steps (I).Catch_Blocks (C).Body_Start := 0;
            Steps.Steps (I).Catch_Blocks (C).Body_Count := 0;
         end loop;
      end loop;
   end Init_Step_Collection;

end STUNIR_Types;
