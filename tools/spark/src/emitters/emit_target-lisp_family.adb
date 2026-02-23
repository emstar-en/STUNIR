--  Emit Target Lisp Family - Lisp Dialect Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Identifier_Strings;
use Identifier_Strings;

package body Emit_Target.Lisp_Family is

   function Is_In_Nested_Block (Func : IR_Function; Idx : Step_Index) return Boolean is
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

   procedure Emit_Steps_Clojure (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  ;; TODO: implement");
         Append_Line ("  nil");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Clojure;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
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
                  if Step.Else_Count > 0 then
                     Append_Line ("  (if " & Cond);
                     Append_Line ("    (do");
                     for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
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
                  Append_Line ("  ;; for loop: use loop/recur");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
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

   procedure Emit_Steps_Common_Lisp (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0)  ; TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Common_Lisp;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
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
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
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
         <<Continue_Common_Lisp>>
      end loop;
   end Emit_Steps_Common_Lisp;

   procedure Emit_Steps_Scheme (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0)  ; TODO: implement");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Scheme;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
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

end Emit_Target.Lisp_Family;