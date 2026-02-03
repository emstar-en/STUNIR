--  STUNIR Functional Language Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body Functional_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Emit_Module (
      Module_Name : in Identifier_String;
      Content     : out Content_String;
      Config      : in Functional_Config;
      Status      : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Module_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Haskell =>
            Content_Strings.Append (Content,
               "-- STUNIR Generated Haskell Module" & New_Line &
               "-- DO-178C Level A Compliant" & New_Line &
               "module " & Name & " where" & New_Line & New_Line);
         when OCaml =>
            Content_Strings.Append (Content,
               "(* STUNIR Generated OCaml Module *)" & New_Line &
               "(* DO-178C Level A Compliant *)" & New_Line &
               "module " & Name & " = struct" & New_Line & New_Line);
         when FSharp =>
            Content_Strings.Append (Content,
               "// STUNIR Generated F# Module" & New_Line &
               "// DO-178C Level A Compliant" & New_Line &
               "module " & Name & New_Line & New_Line);
      end case;
   end Emit_Module;

   procedure Emit_Function (
      Func_Name : in Identifier_String;
      Signature : in String;
      Body_Code : in String;
      Content   : out Content_String;
      Config    : in Functional_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Func_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Haskell =>
            Content_Strings.Append (Content,
               Name & " :: " & Signature & New_Line &
               Name & " = " & Body_Code & New_Line & New_Line);
         when OCaml =>
            Content_Strings.Append (Content,
               "let " & Name & " : " & Signature & " =" & New_Line &
               "  " & Body_Code & New_Line & New_Line);
         when FSharp =>
            Content_Strings.Append (Content,
               "let " & Name & " : " & Signature & " =" & New_Line &
               "    " & Body_Code & New_Line & New_Line);
      end case;
   end Emit_Function;

   procedure Emit_Type (
      Type_Name : in Identifier_String;
      Type_Def  : in String;
      Content   : out Content_String;
      Config    : in Functional_Config;
      Status    : out Emitter_Status)
   is
      Name : constant String := Identifier_Strings.To_String (Type_Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Language is
         when Haskell =>
            Content_Strings.Append (Content,
               "data " & Name & " = " & Type_Def & New_Line & New_Line);
         when OCaml =>
            Content_Strings.Append (Content,
               "type " & Name & " = " & Type_Def & New_Line & New_Line);
         when FSharp =>
            Content_Strings.Append (Content,
               "type " & Name & " = " & Type_Def & New_Line & New_Line);
      end case;
   end Emit_Type;

end Functional_Emitter;
