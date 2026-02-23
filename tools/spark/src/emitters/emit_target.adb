--  Emit Target Micro-Tool Body
--  Emits target language code from IR
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with IR_Parse;
with Ada.Characters.Latin_1;
with Ada.Text_IO;
with Emit_Target.Mainstream;

package body Emit_Target is

   LF : constant Character := Ada.Characters.Latin_1.LF;

   --  Internal helper to append to code string
   procedure Append_Code
     (Code   : in out Code_String;
      Text   : in     String;
      Status :    out Status_Code)
   is
   begin
      if Code_Strings.Length (Code) + Text'Length > Max_Code_Length then
         Status := Error_Too_Large;
         return;
      end if;
      Code_Strings.Append (Code, Text);
      Status := Success;
   end Append_Code;

   function Get_Target_Extension (Target : Target_Language) return String is
   begin
      case Target is
         when Target_CPP       => return ".cpp";
         when Target_C         => return ".c";
         when Target_Python    => return ".py";
         when Target_Rust      => return ".rs";
         when Target_Go        => return ".go";
         when Target_Java      => return ".java";
         when Target_JavaScript => return ".js";
         when Target_CSharp    => return ".cs";
         when Target_Swift     => return ".swift";
         when Target_Kotlin    => return ".kt";
         when Target_SPARK     => return ".adb";
         when Target_Ada       => return ".adb";
         --  Lisp family
         when Target_Common_Lisp   => return ".lisp";
         when Target_Scheme        => return ".scm";
         when Target_Racket        => return ".rkt";
         when Target_Emacs_Lisp    => return ".el";
         when Target_Guile         => return ".scm";
         when Target_Hy            => return ".hy";
         when Target_Janet         => return ".janet";
         when Target_Clojure       => return ".clj";
         when Target_ClojureScript => return ".cljs";
         --  Prolog family
         when Target_SWI_Prolog  => return ".pl";
         when Target_GNU_Prolog => return ".pl";
         when Target_Mercury    => return ".m";
         when Target_Prolog     => return ".pl";
         --  Functional/Formal
         when Target_Futhark   => return ".fut";
         when Target_Lean4     => return ".lean";
         when Target_Haskell   => return ".hs";
      end case;
   end Get_Target_Extension;

   procedure Emit_Single_Target
     (IR       : in     IR_Data;
      Target   : in     Target_Language;
      Code     :    out Code_String;
      Status   :    out Status_Code)
   is
      Temp_Status : Status_Code;
      procedure Append_Line (Text : String) is
      begin
         Append_Code (Code, Text & LF, Temp_Status);
      end Append_Line;

      procedure Emit_Steps_C (Func : IR_Function) is
         Step : IR_Step;
         
         --  Check if step index is inside a nested block
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  --  Check if step is in then block
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                     --  Check if step is in else block
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
                  --  Check if step is in while/for body
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("    // TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            --  Skip steps that are part of nested blocks
            if Is_In_Nested_Block (J) then
               goto Continue;
            end if;
            
            Step := Func.Steps.Steps (J);
            declare
               Val : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
               Init : constant String := Identifier_Strings.To_String (Step.Init);
               Incr : constant String := Identifier_Strings.To_String (Step.Increment);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     if Tgt'Length > 0 then
                        Append_Line ("    " & Tgt & " = " & Val & ";");
                     else
                        Append_Line ("    // assign: " & Val);
                     end if;
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("    " & Tgt & " = " & Val & "(" & Args & ");");
                     else
                        Append_Line ("    " & Val & "(" & Args & ");");
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("    return " & Val & ";");
                     else
                        Append_Line ("    return;");
                     end if;
                  when Step_If =>
                     Append_Line ("    if (" & Cond & ") {");
                     --  Emit then block steps
                     for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "();");
                                    else
                                       Append_Line ("        " & B_Val & "();");
                                    end if;
                                 when Step_Return =>
                                    if B_Val'Length > 0 then
                                       Append_Line ("        return " & B_Val & ";");
                                    else
                                       Append_Line ("        return;");
                                    end if;
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                     if Step.Else_Count > 0 then
                        Append_Line ("    } else {");
                        for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("        " & B_Tgt & " = " & B_Val & "();");
                                       else
                                          Append_Line ("        " & B_Val & "();");
                                       end if;
                                    when Step_Return =>
                                       if B_Val'Length > 0 then
                                          Append_Line ("        return " & B_Val & ";");
                                       else
                                          Append_Line ("        return;");
                                       end if;
                                    when others =>
                                       Append_Line ("        // step");
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                     Append_Line ("    }");
                  when Step_While =>
                     Append_Line ("    while (" & Cond & ") {");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "();");
                                    else
                                       Append_Line ("        " & B_Val & "();");
                                    end if;
                                 when Step_Return =>
                                    if B_Val'Length > 0 then
                                       Append_Line ("        return " & B_Val & ";");
                                    else
                                       Append_Line ("        return;");
                                    end if;
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("    }");
                  when Step_For =>
                     Append_Line ("    for (" & Init & "; " & Cond & "; " & Incr & ") {");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "();");
                                    else
                                       Append_Line ("        " & B_Val & "();");
                                    end if;
                                 when Step_Return =>
                                    if B_Val'Length > 0 then
                                       Append_Line ("        return " & B_Val & ";");
                                    else
                                       Append_Line ("        return;");
                                    end if;
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("    }");
                  when Step_Nop =>
                     Append_Line ("    ;");
                  when others =>
                     Append_Line ("    // TODO: unsupported step");
               end case;
            end;
            <<Continue>>
         end loop;
      end Emit_Steps_C;

      procedure Emit_Steps_Clojure (Func : IR_Function) is
         
         --  Check if step index is inside a nested block
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  --  Check if step is in then block
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                     --  Check if step is in else block
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
                  --  Check if step is in while/for body
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  ;; TODO: implement");
            Append_Line ("  nil");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            --  Skip steps that are part of nested blocks
            if Is_In_Nested_Block (J) then
               goto Continue_Clojure;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
               Init : constant String := Identifier_Strings.To_String (Step.Init);
               Incr : constant String := Identifier_Strings.To_String (Step.Increment);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     if Tgt'Length > 0 then
                        Append_Line ("  (set! " & Tgt & " " & Val & ")");
                     else
                        Append_Line ("  ;; assign " & Val);
                     end if;
                  when Step_Call =>
                     Append_Line ("  (" & Val & " " & Args & ")");
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  nil");
                     end if;
                  when Step_If =>
                     --  Build then branch - emit directly
                     if Step.Else_Count > 0 then
                        Append_Line ("  (if " & Cond);
                        Append_Line ("    (do");
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("    (do");
                        for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("  )");
                     else
                        Append_Line ("  (if " & Cond);
                        Append_Line ("    (do");
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("    nil");
                        Append_Line ("  )");
                     end if;
                  when Step_While =>
                     Append_Line ("  (while " & Cond & " (do");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                              B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    (set! " & B_Tgt & " " & B_Val & ")");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                    else
                                       Append_Line ("    (" & B_Val & " " & B_Args & ")");
                                    end if;
                                 when others =>
                                    Append_Line ("    nil");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("  ))");
                  when Step_For =>
                     Append_Line ("  ;; for loop: " & Init & "; " & Cond & "; " & Incr);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  ;;   " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("  ;; TODO: unsupported step");
               end case;
            end;
            <<Continue_Clojure>>
         end loop;
      end Emit_Steps_Clojure;

      procedure Emit_Steps_Futhark (Func : IR_Function) is
         
         --  Check if step index is inside a nested block
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  --  Check if step is in then block
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                     --  Check if step is in else block
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
                  --  Check if step is in while/for body
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  0  -- TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            --  Skip steps that are part of nested blocks
            if Is_In_Nested_Block (J) then
               goto Continue_Futhark;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
               Init : constant String := Identifier_Strings.To_String (Step.Init);
               Incr : constant String := Identifier_Strings.To_String (Step.Increment);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     Append_Line ("  let " & Tgt & " = " & Val);
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("  let " & Tgt & " = " & Val & " " & Args);
                     else
                        Append_Line ("  " & Val & " " & Args);
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  0");
                     end if;
                  when Step_If =>
                     --  Futhark if expression - handle assignments in branches
                     --  Check if both branches have a single assignment to the same target
                     declare
                        Then_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Else_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Assign_Target    : Identifier_String := Identifier_Strings.Null_Bounded_String;
                     begin
                        --  Check then block for single assignment
                        if Step.Then_Count = 1 and then Step.Then_Start <= Func.Steps.Count then
                           declare
                              TS : constant IR_Step := Func.Steps.Steps (Step.Then_Start);
                           begin
                              if TS.Step_Type = Step_Assign then
                                 Then_Assign_Val := TS.Value;
                                 Assign_Target := TS.Target;
                              end if;
                           end;
                        end if;
                        --  Check else block for single assignment to same target
                        if Step.Else_Count = 1 and then Step.Else_Start <= Func.Steps.Count then
                           declare
                              ES : constant IR_Step := Func.Steps.Steps (Step.Else_Start);
                           begin
                              if ES.Step_Type = Step_Assign then
                                 if Identifier_Strings."=" (ES.Target, Assign_Target) then
                                    Else_Assign_Val := ES.Value;
                                 end if;
                              end if;
                           end;
                        end if;
                        --  Emit appropriate if expression
                        if Identifier_Strings.Length (Then_Assign_Val) > 0 and Identifier_Strings.Length (Else_Assign_Val) > 0 then
                           Append_Line ("  let " & Identifier_Strings.To_String (Assign_Target) & " = if " & Cond & " then " & Identifier_Strings.To_String (Then_Assign_Val) & " else " & Identifier_Strings.To_String (Else_Assign_Val));
                        else
                           --  Fallback: emit if expression with return values
                           declare
                              Then_Val : constant String := 
                                (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                     else "0")
                                 else "0");
                              Else_Val : constant String :=
                                (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
                                     else "0")
                                 else "0");
                           begin
                              Append_Line ("  if " & Cond & " then " & Then_Val & " else " & Else_Val);
                           end;
                        end if;
                     end;
                  when Step_While =>
                     --  Futhark uses loop expressions
                     Append_Line ("  -- while " & Cond & " (use loop expression)");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when Step_For =>
                     --  Futhark uses loop expressions
                     Append_Line ("  -- for " & Init & "; " & Cond & "; " & Incr);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("  -- TODO: unsupported step");
               end case;
            end;
            <<Continue_Futhark>>
         end loop;
      end Emit_Steps_Futhark;

      procedure Emit_Steps_Lean4 (Func : IR_Function) is
         
         --  Check if step index is inside a nested block
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  --  Check if step is in then block
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                     --  Check if step is in else block
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
                  --  Check if step is in while/for body
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then
                           return True;
                        end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  0  -- TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            --  Skip steps that are part of nested blocks
            if Is_In_Nested_Block (J) then
               goto Continue_Lean4;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
               Init : constant String := Identifier_Strings.To_String (Step.Init);
               Incr : constant String := Identifier_Strings.To_String (Step.Increment);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     Append_Line ("  let " & Tgt & " := " & Val);
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("  let " & Tgt & " := " & Val & " " & Args);
                     else
                        Append_Line ("  " & Val & " " & Args);
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  by admit");
                     end if;
                  when Step_If =>
                     --  Lean4 if-then-else expression - handle assignments in branches
                     declare
                        Then_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Else_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Assign_Target    : Identifier_String := Identifier_Strings.Null_Bounded_String;
                     begin
                        --  Check then block for single assignment
                        if Step.Then_Count = 1 and then Step.Then_Start <= Func.Steps.Count then
                           declare
                              TS : constant IR_Step := Func.Steps.Steps (Step.Then_Start);
                           begin
                              if TS.Step_Type = Step_Assign then
                                 Then_Assign_Val := TS.Value;
                                 Assign_Target := TS.Target;
                              end if;
                           end;
                        end if;
                        --  Check else block for single assignment to same target
                        if Step.Else_Count = 1 and then Step.Else_Start <= Func.Steps.Count then
                           declare
                              ES : constant IR_Step := Func.Steps.Steps (Step.Else_Start);
                           begin
                              if ES.Step_Type = Step_Assign then
                                 if Identifier_Strings."=" (ES.Target, Assign_Target) then
                                    Else_Assign_Val := ES.Value;
                                 end if;
                              end if;
                           end;
                        end if;
                        --  Emit appropriate if expression
                        if Identifier_Strings.Length (Then_Assign_Val) > 0 and Identifier_Strings.Length (Else_Assign_Val) > 0 then
                           Append_Line ("  let " & Identifier_Strings.To_String (Assign_Target) & " := if " & Cond & " then " & Identifier_Strings.To_String (Then_Assign_Val) & " else " & Identifier_Strings.To_String (Else_Assign_Val));
                        else
                           --  Fallback: emit if expression with return values
                           declare
                              Then_Val : constant String := 
                                (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                     else "by admit")
                                 else "by admit");
                              Else_Val : constant String :=
                                (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
                                     else "by admit")
                                 else "by admit");
                           begin
                              Append_Line ("  if " & Cond & " then " & Then_Val & " else " & Else_Val);
                           end;
                        end if;
                     end;
                  when Step_While =>
                     --  Lean4 uses recursion or forM
                     Append_Line ("  -- while " & Cond & " (use forM or recursion)");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " := " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when Step_For =>
                     --  Lean4 uses forM
                     Append_Line ("  -- for " & Init & "; " & Cond & "; " & Incr);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " := " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("  -- TODO: unsupported step");
               end case;
            end;
            <<Continue_Lean4>>
         end loop;
      end Emit_Steps_Lean4;

      --  Haskell emitter
      procedure Emit_Steps_Haskell (Func : IR_Function) is
         
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  0  -- TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            if Is_In_Nested_Block (J) then
               goto Continue_Haskell;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
               Init : constant String := Identifier_Strings.To_String (Step.Init);
               Incr : constant String := Identifier_Strings.To_String (Step.Increment);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     Append_Line ("  let " & Tgt & " = " & Val);
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("  let " & Tgt & " = " & Val & " " & Args);
                     else
                        Append_Line ("  " & Val & " " & Args);
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  ()");
                     end if;
                  when Step_If =>
                     --  Haskell if-then-else expression
                     declare
                        Then_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Else_Assign_Val : Identifier_String := Identifier_Strings.Null_Bounded_String;
                        Assign_Target    : Identifier_String := Identifier_Strings.Null_Bounded_String;
                     begin
                        --  Check then block for single assignment
                        if Step.Then_Count = 1 and then Step.Then_Start <= Func.Steps.Count then
                           declare
                              TS : constant IR_Step := Func.Steps.Steps (Step.Then_Start);
                           begin
                              if TS.Step_Type = Step_Assign then
                                 Then_Assign_Val := TS.Value;
                                 Assign_Target := TS.Target;
                              end if;
                           end;
                        end if;
                        --  Check else block for single assignment to same target
                        if Step.Else_Count = 1 and then Step.Else_Start <= Func.Steps.Count then
                           declare
                              ES : constant IR_Step := Func.Steps.Steps (Step.Else_Start);
                           begin
                              if ES.Step_Type = Step_Assign then
                                 if Identifier_Strings."=" (ES.Target, Assign_Target) then
                                    Else_Assign_Val := ES.Value;
                                 end if;
                              end if;
                           end;
                        end if;
                        --  Emit appropriate if expression
                        if Identifier_Strings.Length (Then_Assign_Val) > 0 and Identifier_Strings.Length (Else_Assign_Val) > 0 then
                           Append_Line ("  let " & Identifier_Strings.To_String (Assign_Target) & " = if " & Cond & " then " & Identifier_Strings.To_String (Then_Assign_Val) & " else " & Identifier_Strings.To_String (Else_Assign_Val));
                        else
                           --  Fallback: emit if expression with return values
                           declare
                              Then_Val : constant String := 
                                (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                     else "()")
                                 else "()");
                              Else_Val : constant String :=
                                (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                    (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                        Identifier_Strings.To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
                                     else "()")
                                 else "()");
                           begin
                              Append_Line ("  if " & Cond & " then " & Then_Val & " else " & Else_Val);
                           end;
                        end if;
                     end;
                  when Step_While =>
                     --  Haskell uses recursion
                     Append_Line ("  -- while " & Cond & " (use recursion)");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when Step_For =>
                     --  Haskell uses mapM_ or forM_
                     Append_Line ("  -- for " & Init & "; " & Cond & "; " & Incr);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  --   let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("  -- TODO: unsupported step");
               end case;
            end;
            <<Continue_Haskell>>
         end loop;
      end Emit_Steps_Haskell;

      --  Common Lisp emitter (supports CLOS, setf, defun, etc.)
      procedure Emit_Steps_Common_Lisp (Func : IR_Function) is
         
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  0)  ; TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            if Is_In_Nested_Block (J) then
               goto Continue_Common_Lisp;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     if Tgt'Length > 0 then
                        Append_Line ("  (setf " & Tgt & " " & Val & ")");
                     end if;
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("  (setf " & Tgt & " (" & Val & " " & Args & "))");
                     else
                        Append_Line ("  (" & Val & " " & Args & ")");
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  nil");
                     end if;
                  when Step_If =>
                     if Step.Else_Count > 0 then
                        Append_Line ("  (if " & Cond);
                        Append_Line ("    (progn");
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (setf " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (setf " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("    (progn");
                        for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (setf " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (setf " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("  )");
                     else
                        Append_Line ("  (when " & Cond);
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("    (setf " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("    (setf " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("    (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("    " & B_Val);
                                    when others =>
                                       Append_Line ("    nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("  )");
                     end if;
                  when Step_While =>
                     Append_Line ("  (loop while " & Cond & " do");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                              B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    (setf " & B_Tgt & " " & B_Val & ")");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    (setf " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                    else
                                       Append_Line ("    (" & B_Val & " " & B_Args & ")");
                                    end if;
                                 when others =>
                                    Append_Line ("    nil");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("  )");
                  when Step_For =>
                     Append_Line ("  ;; for loop: use loop macro");
                     Append_Line ("  (loop for i from 1 to n do");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    (setf " & B_Tgt & " " & B_Val & ")");
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("  )");
                  when others =>
                     Append_Line ("  ;; TODO: unsupported step");
               end case;
            end;
            <<Continue_Common_Lisp>>
         end loop;
      end Emit_Steps_Common_Lisp;

      --  Scheme/Racket/Guile emitter (R5RS/R6RS/Racket compatible)
      procedure Emit_Steps_Scheme (Func : IR_Function) is
         
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("  0)  ; TODO: implement");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            if Is_In_Nested_Block (J) then
               goto Continue_Scheme;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     if Tgt'Length > 0 then
                        Append_Line ("  (set! " & Tgt & " " & Val & ")");
                     end if;
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("  (set! " & Tgt & " (" & Val & " " & Args & "))");
                     else
                        Append_Line ("  (" & Val & " " & Args & ")");
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("  " & Val);
                     else
                        Append_Line ("  #f");
                     end if;
                  when Step_If =>
                     if Step.Else_Count > 0 then
                        Append_Line ("  (if " & Cond);
                        Append_Line ("    (begin");
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      #f");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("    (begin");
                        for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      #f");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                        Append_Line ("  )");
                     else
                        Append_Line ("  (when " & Cond);
                        for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("    (set! " & B_Tgt & " " & B_Val & ")");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("    (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                       else
                                          Append_Line ("    (" & B_Val & " " & B_Args & ")");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("    " & B_Val);
                                    when others =>
                                       Append_Line ("    #f");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("  )");
                     end if;
                  when Step_While =>
                     Append_Line ("  (let loop ()");
                     Append_Line ("    (when " & Cond);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                              B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      (set! " & B_Tgt & " (" & B_Val & " " & B_Args & "))");
                                    else
                                       Append_Line ("      (" & B_Val & " " & B_Args & ")");
                                    end if;
                                 when others =>
                                    Append_Line ("      #f");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("      (loop))");
                     Append_Line ("    )");
                     Append_Line ("  )");
                  when Step_For =>
                     Append_Line ("  ;; for loop: use named let or do");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("  ;   (set! " & B_Tgt & " " & B_Val & ")");
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("  ; TODO: unsupported step");
               end case;
            end;
            <<Continue_Scheme>>
         end loop;
      end Emit_Steps_Scheme;

      --  Prolog emitter (SWI/GNU/Mercury compatible)
      procedure Emit_Steps_Prolog (Func : IR_Function) is
         
         function Is_In_Nested_Block (Idx : Step_Index) return Boolean is
         begin
            for J in Step_Index range 1 .. Func.Steps.Count loop
               declare
                  S : constant IR_Step := Func.Steps.Steps (J);
               begin
                  if S.Step_Type = Step_If then
                     for B in Step_Index range S.Then_Start .. S.Then_Start + S.Then_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                     for B in Step_Index range S.Else_Start .. S.Else_Start + S.Else_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
                  if S.Step_Type = Step_While or S.Step_Type = Step_For then
                     for B in Step_Index range S.Body_Start .. S.Body_Start + S.Body_Count - 1 loop
                        if B = Idx then return True; end if;
                     end loop;
                  end if;
               end;
            end loop;
            return False;
         end Is_In_Nested_Block;
      begin
         if Func.Steps.Count = 0 then
            Append_Line ("    % TODO: implement");
            Append_Line ("    Result = nil.");
            return;
         end if;

         for J in Step_Index range 1 .. Func.Steps.Count loop
            if Is_In_Nested_Block (J) then
               goto Continue_Prolog;
            end if;
            
            declare
               Step : constant IR_Step := Func.Steps.Steps (J);
               Val  : constant String := Identifier_Strings.To_String (Step.Value);
               Tgt  : constant String := Identifier_Strings.To_String (Step.Target);
               Cond : constant String := Identifier_Strings.To_String (Step.Condition);
               Args : constant String := Identifier_Strings.To_String (Step.Args);
            begin
               case Step.Step_Type is
                  when Step_Assign =>
                     if Tgt'Length > 0 then
                        Append_Line ("    " & Tgt & " is " & Val & ",");
                     end if;
                  when Step_Call =>
                     if Tgt'Length > 0 then
                        Append_Line ("    " & Val & "(" & Args & ", " & Tgt & "),");
                     else
                        Append_Line ("    " & Val & "(" & Args & "),");
                     end if;
                  when Step_Return =>
                     if Val'Length > 0 then
                        Append_Line ("    Result = " & Val & ".");
                     else
                        Append_Line ("    Result = nil.");
                     end if;
                  when Step_If =>
                     Append_Line ("    (" & Cond & " ->");
                     for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                              B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " is " & B_Val & ",");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Val & "(" & B_Args & ", " & B_Tgt & "),");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & "),");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        Result = " & B_Val);
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                     if Step.Else_Count > 0 then
                        Append_Line ("    ; ");
                        for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                                 B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                                 B_Args : constant String := Identifier_Strings.To_String (Body_Step.Args);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("        " & B_Tgt & " is " & B_Val & ",");
                                       end if;
                                    when Step_Call =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("        " & B_Val & "(" & B_Args & ", " & B_Tgt & "),");
                                       else
                                          Append_Line ("        " & B_Val & "(" & B_Args & "),");
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("        Result = " & B_Val);
                                    when others =>
                                       null;
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                     Append_Line ("    ),");
                  when Step_While =>
                     Append_Line ("    % while loop: use recursion");
                     Append_Line ("    % condition: " & Cond);
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    %   " & B_Tgt & " is " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when Step_For =>
                     Append_Line ("    % for loop: use recursion or between/3");
                     for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := Identifier_Strings.To_String (Body_Step.Value);
                              B_Tgt : constant String := Identifier_Strings.To_String (Body_Step.Target);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("    %   " & B_Tgt & " is " & B_Val);
                                    end if;
                                 when others =>
                                    null;
                              end case;
                           end;
                        end if;
                     end loop;
                  when others =>
                     Append_Line ("    % TODO: unsupported step");
               end case;
            end;
            <<Continue_Prolog>>
         end loop;
      end Emit_Steps_Prolog;

   begin
      Code := Code_Strings.Null_Bounded_String;
      Status := Success;

      --  Emit header comment
      case Target is
         when Target_Python =>
            Append_Code (Code, "# Generated by STUNIR" & LF, Temp_Status);
            Append_Code (Code, "# Module: " & Identifier_Strings.To_String (IR.Module_Name) & LF, Temp_Status);
         when Target_Clojure | Target_ClojureScript | Target_Common_Lisp | Target_Scheme | Target_Racket | Target_Emacs_Lisp | Target_Guile | Target_Hy | Target_Janet =>
            Append_Code (Code, ";; Generated by STUNIR" & LF, Temp_Status);
            Append_Code (Code, ";; Module: " & Identifier_Strings.To_String (IR.Module_Name) & LF, Temp_Status);
         when Target_Futhark | Target_Lean4 =>
            Append_Code (Code, "-- Generated by STUNIR" & LF, Temp_Status);
            Append_Code (Code, "-- Module: " & Identifier_Strings.To_String (IR.Module_Name) & LF, Temp_Status);
         when Target_SWI_Prolog | Target_GNU_Prolog | Target_Mercury | Target_Prolog =>
            Append_Code (Code, "% Generated by STUNIR" & LF, Temp_Status);
            Append_Code (Code, "% Module: " & Identifier_Strings.To_String (IR.Module_Name) & LF, Temp_Status);
         when others =>
            Append_Code (Code, "// Generated by STUNIR" & LF, Temp_Status);
            Append_Code (Code, "// Module: " & Identifier_Strings.To_String (IR.Module_Name) & LF, Temp_Status);
      end case;

      if Temp_Status /= Success then
         Status := Temp_Status;
         return;
      end if;

      --  Emit imports
      if IR.Imports.Count > 0 then
         case Target is
            when Target_Python =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                     Imp_From : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).From_Module);
                  begin
                     if Imp_From'Length > 0 then
                        Append_Code (Code, "from " & Imp_From & " import " & Imp_Name & LF, Temp_Status);
                     else
                        Append_Code (Code, "import " & Imp_Name & LF, Temp_Status);
                     end if;
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_C | Target_CPP =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "#include <" & Imp_Name & ".h>" & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Rust =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "use " & Imp_Name & ";" & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Clojure | Target_ClojureScript =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "(:require [" & Imp_Name & "])" & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Futhark =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "import " & Imp_Name & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Lean4 =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "import " & Imp_Name & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_SWI_Prolog | Target_GNU_Prolog | Target_Prolog =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, ":- use_module(" & Imp_Name & ")." & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Ada | Target_SPARK =>
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "with " & Imp_Name & ";" & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when others =>
               --  Generic comment for unsupported targets
               for I in Import_Index range 1 .. IR.Imports.Count loop
                  declare
                     Imp_Name : constant String := Identifier_Strings.To_String (IR.Imports.Imports (I).Name);
                  begin
                     Append_Code (Code, "// import: " & Imp_Name & LF, Temp_Status);
                  end;
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
         end case;
      end if;

      --  Emit type definitions
      if IR.Types.Count > 0 then
         case Target is
            when Target_C =>
               for I in Type_Def_Index range 1 .. IR.Types.Count loop
                  declare
                     Type_Name : constant String := Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name);
                  begin
                     case IR.Types.Type_Defs (I).Kind is
                        when Type_Struct =>
                           Append_Code (Code, "typedef struct {" & LF, Temp_Status);
                           for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
                              Append_Code (Code, "    " & 
                                 Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & " " &
                                 Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & ";" & LF, Temp_Status);
                           end loop;
                           Append_Code (Code, "} " & Type_Name & ";" & LF & LF, Temp_Status);
                        when Type_Enum =>
                           Append_Code (Code, "typedef enum { /* TODO: enum values */ } " & Type_Name & ";" & LF & LF, Temp_Status);
                        when Type_Alias =>
                           Append_Code (Code, "typedef " & 
                              Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Base_Type) & " " & Type_Name & ";" & LF & LF, Temp_Status);
                        when Type_Generic =>
                           Append_Code (Code, "// TODO: generic type " & Type_Name & LF & LF, Temp_Status);
                     end case;
                  end;
               end loop;
            when Target_Rust =>
               for I in Type_Def_Index range 1 .. IR.Types.Count loop
                  declare
                     Type_Name : constant String := Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name);
                  begin
                     case IR.Types.Type_Defs (I).Kind is
                        when Type_Struct =>
                           Append_Code (Code, "struct " & Type_Name & " {" & LF, Temp_Status);
                           for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
                              Append_Code (Code, "    " & 
                                 Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & ": " &
                                 Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & "," & LF, Temp_Status);
                           end loop;
                           Append_Code (Code, "}" & LF & LF, Temp_Status);
                        when Type_Enum =>
                           Append_Code (Code, "enum " & Type_Name & " { /* TODO: variants */ }" & LF & LF, Temp_Status);
                        when Type_Alias =>
                           Append_Code (Code, "type " & Type_Name & " = " & 
                              Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Base_Type) & ";" & LF & LF, Temp_Status);
                        when Type_Generic =>
                           Append_Code (Code, "// TODO: generic type " & Type_Name & LF & LF, Temp_Status);
                     end case;
                  end;
               end loop;
            when Target_Python =>
               for I in Type_Def_Index range 1 .. IR.Types.Count loop
                  declare
                     Type_Name : constant String := Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name);
                  begin
                     case IR.Types.Type_Defs (I).Kind is
                        when Type_Struct =>
                           Append_Code (Code, "@dataclass" & LF, Temp_Status);
                           Append_Code (Code, "class " & Type_Name & ":" & LF, Temp_Status);
                           if IR.Types.Type_Defs (I).Fields.Count = 0 then
                              Append_Code (Code, "    pass" & LF & LF, Temp_Status);
                           else
                              for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
                                 Append_Code (Code, "    " & 
                                    Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & ": " &
                                    Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & LF, Temp_Status);
                              end loop;
                           end if;
                           Append_Code (Code, "" & LF, Temp_Status);
                        when others =>
                           Append_Code (Code, "# TODO: type " & Type_Name & LF & LF, Temp_Status);
                     end case;
                  end;
               end loop;
            when Target_Ada | Target_SPARK =>
               for I in Type_Def_Index range 1 .. IR.Types.Count loop
                  declare
                     Type_Name : constant String := Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name);
                  begin
                     case IR.Types.Type_Defs (I).Kind is
                        when Type_Struct =>
                           Append_Code (Code, "type " & Type_Name & " is record" & LF, Temp_Status);
                           for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
                              Append_Code (Code, "      " & 
                                 Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & " : " &
                                 Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & ";" & LF, Temp_Status);
                           end loop;
                           Append_Code (Code, "end record;" & LF & LF, Temp_Status);
                        when Type_Enum =>
                           Append_Code (Code, "type " & Type_Name & " is (/* TODO: enum values */);" & LF & LF, Temp_Status);
                        when Type_Alias =>
                           Append_Code (Code, "subtype " & Type_Name & " is " & 
                              Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Base_Type) & ";" & LF & LF, Temp_Status);
                        when Type_Generic =>
                           Append_Code (Code, "-- TODO: generic type " & Type_Name & LF & LF, Temp_Status);
                     end case;
                  end;
               end loop;
            when others =>
               --  Generic comment for unsupported targets
               for I in Type_Def_Index range 1 .. IR.Types.Count loop
                  Append_Code (Code, "// type: " & 
                     Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name) & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
         end case;
      end if;

      --  Emit constants
      if IR.Constants.Count > 0 then
         case Target is
            when Target_C | Target_CPP =>
               for I in Constant_Index range 1 .. IR.Constants.Count loop
                  Append_Code (Code, "#define " & 
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & " " &
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Rust =>
               for I in Constant_Index range 1 .. IR.Constants.Count loop
                  Append_Code (Code, "const " & 
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & ": " &
                     Type_Name_Strings.To_String (IR.Constants.Constants (I).Const_Type) & " = " &
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & ";" & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Python =>
               for I in Constant_Index range 1 .. IR.Constants.Count loop
                  Append_Code (Code, 
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & " = " &
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when Target_Ada | Target_SPARK =>
               for I in Constant_Index range 1 .. IR.Constants.Count loop
                  Append_Code (Code, 
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & " : constant " &
                     Type_Name_Strings.To_String (IR.Constants.Constants (I).Const_Type) & " := " &
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & ";" & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
            when others =>
               for I in Constant_Index range 1 .. IR.Constants.Count loop
                  Append_Code (Code, "// const: " & 
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Name) & " = " &
                     Identifier_Strings.To_String (IR.Constants.Constants (I).Value_Str) & LF, Temp_Status);
               end loop;
               Append_Code (Code, "" & LF, Temp_Status);
         end case;
      end if;

      --  Emit dependencies as comments
      if IR.Dependencies.Count > 0 then
         Append_Code (Code, "// Dependencies:" & LF, Temp_Status);
         for I in Dependency_Index range 1 .. IR.Dependencies.Count loop
            declare
               Dep_Name : constant String := Identifier_Strings.To_String (IR.Dependencies.Dependencies (I).Name);
               Dep_Ver  : constant String := Identifier_Strings.To_String (IR.Dependencies.Dependencies (I).Version);
            begin
               if Dep_Ver'Length > 0 then
                  Append_Code (Code, "//   " & Dep_Name & " @ " & Dep_Ver & LF, Temp_Status);
               else
                  Append_Code (Code, "//   " & Dep_Name & LF, Temp_Status);
               end if;
            end;
         end loop;
         Append_Code (Code, "" & LF, Temp_Status);
      end if;

      --  Emit function stubs
      for I in Function_Index range 1 .. IR.Functions.Count loop
         declare
            Func : constant IR_Function := IR.Functions.Functions (I);
            Func_Name : constant String := Identifier_Strings.To_String (Func.Name);
            Ret_Type  : constant String := Type_Name_Strings.To_String (Func.Return_Type);
         begin
            case Target is
               when Target_Python =>
                  Append_Code (Code, "def " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, "):" & LF, Temp_Status);
                  Append_Code (Code, "    pass  # TODO: implement" & LF & LF, Temp_Status);
                  
               when Target_C | Target_CPP =>
                  Append_Code (Code, Ret_Type & " " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type) & " " &
                                  Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") {" & LF, Temp_Status);
                  Emit_Steps_C (Func);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_Rust =>
                  Append_Code (Code, "fn " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & ": " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") -> " & Ret_Type & " {" & LF, Temp_Status);
                  Append_Code (Code, "    todo!()" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_Go =>
                  Append_Code (Code, "func " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & " " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") " & Ret_Type & " {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_Java =>
                  Append_Code (Code, "public static " & Ret_Type & " " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type) & " " &
                                  Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_JavaScript =>
                  Append_Code (Code, "function " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_CSharp =>
                  Append_Code (Code, "public static " & Ret_Type & " " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type) & " " &
                                  Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_Clojure | Target_ClojureScript =>
                  Append_Code (Code, "(defn " & Func_Name & " [", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, "]" & LF, Temp_Status);
                  Emit_Steps_Clojure (Func);
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               --  Lisp family: Common Lisp
               when Target_Common_Lisp =>
                  Append_Code (Code, "(defun " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ")" & LF, Temp_Status);
                  Emit_Steps_Common_Lisp (Func);
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               --  Lisp family: Scheme/Racket/Guile
               when Target_Scheme | Target_Racket | Target_Guile =>
                  Append_Code (Code, "(define (" & Func_Name & " ", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ")" & LF, Temp_Status);
                  Emit_Steps_Scheme (Func);
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               --  Lisp family: Emacs Lisp
               when Target_Emacs_Lisp =>
                  Append_Code (Code, "(defun " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ")" & LF, Temp_Status);
                  Emit_Steps_Common_Lisp (Func);  --  Emacs Lisp is similar to Common Lisp
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               --  Lisp family: Hy (Python Lisp)
               when Target_Hy =>
                  Append_Code (Code, "(defn " & Func_Name & " [", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, "]" & LF, Temp_Status);
                  Emit_Steps_Clojure (Func);  --  Hy is similar to Clojure
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               --  Lisp family: Janet
               when Target_Janet =>
                  Append_Code (Code, "(defn " & Func_Name & " [", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, " ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, "]" & LF, Temp_Status);
                  Emit_Steps_Clojure (Func);  --  Janet is similar to Clojure
                  Append_Code (Code, ")" & LF & LF, Temp_Status);
                  
               when Target_Futhark =>
                  Append_Code (Code, "let " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & ": " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ") (", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") : " & Ret_Type & " =" & LF, Temp_Status);
                  Emit_Steps_Futhark (Func);
                  Append_Code (Code, "" & LF, Temp_Status);
                  
               when Target_Lean4 =>
                  Append_Code (Code, "def " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & " : " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ") (", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") : " & Ret_Type & " :=" & LF, Temp_Status);
                  Emit_Steps_Lean4 (Func);
                  Append_Code (Code, "" & LF, Temp_Status);
                  
               when Target_Haskell =>
                  Append_Code (Code, Func_Name & " :: " & Ret_Type & LF, Temp_Status);
                  Append_Code (Code, Func_Name & " ", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & " ", Temp_Status);
                  end loop;
                  Append_Code (Code, "=" & LF, Temp_Status);
                  Emit_Steps_Haskell (Func);
                  Append_Code (Code, "" & LF, Temp_Status);
                  
               --  Prolog family: SWI/GNU/Mercury/Generic
               when Target_SWI_Prolog | Target_GNU_Prolog | Target_Mercury | Target_Prolog =>
                  Append_Code (Code, "% " & Func_Name & "(" & Ret_Type & ")" & LF, Temp_Status);
                  Append_Code (Code, Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ", Result) :-" & LF, Temp_Status);
                  Emit_Steps_Prolog (Func);
                  Append_Code (Code, "" & LF, Temp_Status);
                  
               when Target_SPARK =>
                  Append_Code (Code, "function " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & " : " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, "; ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") return " & Ret_Type & " is" & LF, Temp_Status);
                  Append_Code (Code, "begin" & LF, Temp_Status);
                  Mainstream.Emit_Steps_SPARK (Func, Append_Line'Access);
                  Append_Code (Code, "end " & Func_Name & ";" & LF & LF, Temp_Status);
                  
               when Target_Ada =>
                  Append_Code (Code, "function " & Func_Name & " (", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & " : " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, "; ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") return " & Ret_Type & " is" & LF, Temp_Status);
                  Append_Code (Code, "begin" & LF, Temp_Status);
                  Mainstream.Emit_Steps_Ada (Func, Append_Line'Access);
                  Append_Code (Code, "end " & Func_Name & ";" & LF & LF, Temp_Status);
                  
               when Target_Swift =>
                  Append_Code (Code, "func " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & ": " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, ") -> " & Ret_Type & " {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
                  
               when Target_Kotlin =>
                  Append_Code (Code, "fun " & Func_Name & "(", Temp_Status);
                  for J in Parameter_Index range 1 .. Func.Parameters.Count loop
                     Append_Code (Code, Identifier_Strings.To_String (Func.Parameters.Params (J).Name) & ": " &
                                  Type_Name_Strings.To_String (Func.Parameters.Params (J).Param_Type), Temp_Status);
                     if J < Func.Parameters.Count then
                        Append_Code (Code, ", ", Temp_Status);
                     end if;
                  end loop;
                  Append_Code (Code, "): " & Ret_Type & " {" & LF, Temp_Status);
                  Append_Code (Code, "    // TODO: implement" & LF, Temp_Status);
                  Append_Code (Code, "}" & LF & LF, Temp_Status);
            end case;
            
            if Temp_Status /= Success then
               Status := Temp_Status;
               return;
            end if;
         end;
      end loop;

      --  Emit footer
      case Target is
         when Target_Python =>
            Append_Code (Code, "# End of module" & LF, Temp_Status);
         when Target_Clojure | Target_ClojureScript | Target_Common_Lisp | Target_Scheme | Target_Racket | Target_Emacs_Lisp | Target_Guile | Target_Hy | Target_Janet =>
            Append_Code (Code, ";; End of module" & LF, Temp_Status);
         when Target_Futhark | Target_Lean4 | Target_Haskell =>
            Append_Code (Code, "-- End of module" & LF, Temp_Status);
         when Target_SWI_Prolog | Target_GNU_Prolog | Target_Mercury | Target_Prolog =>
            Append_Code (Code, "% End of module" & LF, Temp_Status);
         when others =>
            Append_Code (Code, "// End of module" & LF, Temp_Status);
      end case;
   end Emit_Single_Target;

   procedure Emit_Target_File
     (Input_Path  : in     Path_String;
      Target      : in     Target_Language;
      Output_Path : in     Path_String;
      Status      :    out Status_Code)
   is
      pragma SPARK_Mode (Off);  --  File I/O not in SPARK
      
      use Ada.Text_IO;
      
      IR   : IR_Data;
      Code : Code_String;
   begin
      --  Parse IR file
      IR_Parse.Parse_IR_File (Input_Path, IR, Status);
      if Status /= Success then
         return;
      end if;
      
      --  Emit code
      Emit_Single_Target (IR, Target, Code, Status);
      if Status /= Success then
         return;
      end if;
      
      --  Write output file
      declare
         Output_File : File_Type;
      begin
         Create (Output_File, Out_File, Path_Strings.To_String (Output_Path));
         Put (Output_File, Code_Strings.To_String (Code));
         Close (Output_File);
      exception
         when others =>
            Status := Error_File_IO;
            return;
      end;
      
      Status := Success;
   end Emit_Target_File;

end Emit_Target;
