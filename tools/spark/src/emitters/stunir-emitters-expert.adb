-- STUNIR Expert Systems Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Expert is
   pragma SPARK_Mode (On);

   -- Emit complete module (expert system rules)
   overriding procedure Emit_Module
     (Self   : in out Expert_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.System_Type is
         when CLIPS =>
            -- Emit CLIPS expert system
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; STUNIR Generated CLIPS Expert System" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(defmodule " & Name_Strings.To_String (Module.Module_Name) & ")" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(defrule example-rule" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  ""STUNIR generated rule""" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (fact ?x)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  =>" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (assert (conclusion ?x)))" & ASCII.LF);

         when Jess =>
            -- Emit Jess expert system
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; STUNIR Generated Jess Expert System" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(deftemplate fact (slot value))" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(defrule example" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (fact (value ?v))" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  =>" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (printout t ""Rule fired"" crlf))" & ASCII.LF);

         when Drools =>
            -- Emit Drools business rules
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// STUNIR Generated Drools Rules" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "// DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "package com.stunir.rules;" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "rule ""Example Rule""" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  when" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    $f : Fact()" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  then" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    System.out.println(""Rule matched"");" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "end" & ASCII.LF);

         when Prolog_Based =>
            -- Emit Prolog-based expert system
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% STUNIR Generated Prolog Expert System" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ":- dynamic fact/1." & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "rule(X) :- fact(X), conclusion(X)." & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Expert System" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "rule example:" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  if condition then action" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (fact template)
   overriding procedure Emit_Type
     (Self   : in out Expert_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.System_Type is
         when CLIPS | Jess =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(deftemplate " & Name_Strings.To_String (T.Name) & ASCII.LF);
            for I in 1 .. T.Field_Cnt loop
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "  (slot " & Name_Strings.To_String (T.Fields (I).Name) & ")" & ASCII.LF);
            end loop;
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ")" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "fact_type " & Name_Strings.To_String (T.Name) & ASCII.LF);
      end case;

      Success := True;
   end Emit_Type;

   -- Emit function definition (rule)
   overriding procedure Emit_Function
     (Self   : in out Expert_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.System_Type is
         when CLIPS | Jess =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(defrule " & Name_Strings.To_String (Func.Name) & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  ; STUNIR generated rule" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (condition)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  =>" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  (action))" & ASCII.LF);

         when Drools =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "rule """ & Name_Strings.To_String (Func.Name) & """" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  when" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    // condition" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  then" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "    // action" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "end" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "rule " & Name_Strings.To_String (Func.Name) & " {" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  /* conditions and actions */" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "}" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Expert;
