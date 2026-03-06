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
            Idx  : constant String := To_String (Step.Index);
            Key  : constant String := To_String (Step.Key);
            Field : constant String := To_String (Step.Field);
            Type_Args : constant String := To_String (Step.Type_Args);
            Field_Kw : constant String := (if Field'Length > 0 then ":" & Field else ":field");
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
                  Append_Line ("  (loop []");
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
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("      (recur))");
                  Append_Line ("    ))");
               when Step_For =>
                  Append_Line ("  (loop [i " & Init "]");
                  Append_Line ("    (when (< i " & Cond & ")");
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
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("      (recur (inc i))))");
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  (throw (Exception. " & Val & "))");
                  else
                     Append_Line ("  (throw (Exception. \"runtime error\"))");
                  end if;
               when Step_Break =>
                  Append_Line ("  (break)");
               when Step_Continue =>
                  Append_Line ("  (continue)");
               when Step_Switch =>
                  Append_Line ("  (case " & Val);
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    " & Step.Case_Values (C) & " (do");
                        for B in Step_Index range Step.Case_Starts (C) .. Step.Case_Starts (C) + Step.Case_Counts (C) - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                                 B_Tgt : constant String := To_String (Body_Step.Target);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
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
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    :default (do");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("      " & B_Val);
                                 when others =>
                                    Append_Line ("      nil");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("    )");
                  end if;
                  Append_Line ("  )");
               when Step_Try =>
                  Append_Line ("  (try");
                  for B in Step_Index range Step.Try_Start .. Step.Try_Start + Step.Try_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("    (set! " & B_Tgt & " " & B_Val & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("    " & B_Val);
                              when others =>
                                 Append_Line ("    nil");
                           end case;
                        end;
                     end if;
                  end loop;
                  for C in Step_Index range 1 .. Step.Catch_Count loop
                     if Step.Catch_Blocks (C).Exception_Type'Length > 0 then
                        Append_Line ("  (catch " & Step.Catch_Blocks (C).Exception_Type & " e");
                        for B in Step_Index range Step.Catch_Blocks (C).Handler_Start .. Step.Catch_Blocks (C).Handler_Start + Step.Catch_Blocks (C).Handler_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                              begin
                                 case Body_Step.Step_Type is
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
                  end loop;
                  if Step.Finally_Count > 0 then
                     Append_Line ("  (finally");
                     for B in Step_Index range Step.Finally_Start .. Step.Finally_Start + Step.Finally_Count - 1 loop
                        if B <= Func.Steps.Count then
                           Append_Line ("    nil");
                        end if;
                     end loop;
                     Append_Line ("  )");
                  end if;
                  Append_Line ("  )");
               when Step_Throw =>
                  Append_Line ("  (throw (Exception. " & Val & "))");
               when Step_Array_New =>
                  Append_Line ("  (set! " & Tgt & " (into-array []))");
               when Step_Array_Get =>
                  Append_Line ("  (set! " & Tgt & " (aget " & Val & " " & Idx & "))");
               when Step_Array_Set =>
                  Append_Line ("  (aset " & Tgt & " " & Idx & " " & Val & ")");
               when Step_Array_Push =>
                  Append_Line ("  (set! " & Tgt & " (conj " & Tgt & " " & Val & "))");
               when Step_Array_Pop =>
                  Append_Line ("  (set! " & Tgt & " (pop " & Val & "))");
               when Step_Array_Len =>
                  Append_Line ("  (set! " & Tgt & " (count " & Val & "))");
               when Step_Map_New =>
                  Append_Line ("  (set! " & Tgt & " {})");
               when Step_Map_Get =>
                  Append_Line ("  (set! " & Tgt & " (get " & Val & " " & Key & "))");
               when Step_Map_Set =>
                  Append_Line ("  (set! " & Tgt & " (assoc " & Tgt & " " & Key & " " & Val & "))");
               when Step_Map_Delete =>
                  Append_Line ("  (set! " & Tgt & " (dissoc " & Val & " " & Key & "))");
               when Step_Map_Has =>
                  Append_Line ("  (set! " & Tgt & " (contains? " & Val & " " & Key & "))");
               when Step_Set_New =>
                  Append_Line ("  (set! " & Tgt & " #{})");
               when Step_Set_Add =>
                  Append_Line ("  (set! " & Tgt & " (conj " & Tgt & " " & Val & "))");
               when Step_Set_Remove =>
                  Append_Line ("  (set! " & Tgt & " (disj " & Tgt & " " & Val & "))");
               when Step_Set_Has =>
                  Append_Line ("  (set! " & Tgt & " (contains? " & Args & " " & Val & "))");
               when Step_Struct_New =>
                  Append_Line ("  (set! " & Tgt & " (->" & Val & "))");
               when Step_Struct_Get =>
                  Append_Line ("  (set! " & Tgt & " (:" & Field & " " & Val & "))");
               when Step_Struct_Set =>
                  Append_Line ("  (set! " & Tgt & " (assoc " & Tgt & " :" & Field & " " & Val & "))");
               when Step_Generic_Call =>
                  Append_Line ("  (set! " & Tgt & " (" & Val & " " & Type_Args & " " & Args & "))");
               when Step_Type_Cast =>
                  Append_Line ("  (set! " & Tgt & " (" & Args & " " & Val & "))");
               when others =>
                    Append_Line ("  ;; unsupported step");
            end case;
         end;
         <<Continue_Clojure>>
      end loop;
   end Emit_Steps_Clojure;

   procedure Emit_Steps_Common_Lisp (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0)  ; no steps");
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
                  Append_Line ("  (loop for i from " & Init & " below " & Cond & " do");
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
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  (error " & Val & ")");
                  else
                     Append_Line ("  (error \"runtime error\")");
                  end if;
               when Step_Break =>
                  Append_Line ("  (break)");
               when Step_Continue =>
                  Append_Line ("  (continue)");
               when Step_Switch =>
                  Append_Line ("  (case " & Val);
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    ((" & Step.Case_Values (C) & ")");
                        for B in Step_Index range Step.Case_Starts (C) .. Step.Case_Starts (C) + Step.Case_Counts (C) - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                                 B_Tgt : constant String := To_String (Body_Step.Target);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (setf " & B_Tgt & " " & B_Val & ")");
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
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    (t");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("      " & B_Val);
                                 when others =>
                                    Append_Line ("      nil");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("    )");
                  end if;
                  Append_Line ("  )");
               when Step_Try =>
                  Append_Line ("  (handler-case");
                  Append_Line ("    (progn");
                  for B in Step_Index range Step.Try_Start .. Step.Try_Start + Step.Try_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      (setf " & B_Tgt & " " & B_Val & ")");
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
                  for C in Step_Index range 1 .. Step.Catch_Count loop
                     if Step.Catch_Blocks (C).Exception_Type'Length > 0 then
                        Append_Line ("    (" & Step.Catch_Blocks (C).Exception_Type & " (e)");
                        for B in Step_Index range Step.Catch_Blocks (C).Handler_Start .. Step.Catch_Blocks (C).Handler_Start + Step.Catch_Blocks (C).Handler_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Return =>
                                       Append_Line ("      " & B_Val);
                                    when others =>
                                       Append_Line ("      nil");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("    )");
                     end if;
                  end loop;
                  Append_Line ("  )");
               when Step_Throw =>
                  Append_Line ("  (error " & Val & ")");
               when Step_Array_New =>
                  Append_Line ("  (setf " & Tgt & " (make-array " & Val & "))");
               when Step_Array_Get =>
                  Append_Line ("  (setf " & Tgt & " (aref " & Val & " " & Idx & "))");
               when Step_Array_Set =>
                  Append_Line ("  (setf (aref " & Tgt & " " & Idx & ") " & Val & ")");
               when Step_Array_Push =>
                  Append_Line ("  (vector-push-extend " & Val & " " & Tgt & ")");
               when Step_Array_Pop =>
                  Append_Line ("  (setf " & Tgt & " (vector-pop " & Val & "))");
               when Step_Array_Len =>
                  Append_Line ("  (setf " & Tgt & " (length " & Val & "))");
               when Step_Map_New =>
                  Append_Line ("  (setf " & Tgt & " (make-hash-table))");
               when Step_Map_Get =>
                  Append_Line ("  (setf " & Tgt & " (gethash " & Key & " " & Val & "))");
               when Step_Map_Set =>
                  Append_Line ("  (setf (gethash " & Key & " " & Tgt & ") " & Val & ")");
               when Step_Map_Delete =>
                  Append_Line ("  (remhash " & Key & " " & Val & ")");
               when Step_Map_Has =>
                  Append_Line ("  (setf " & Tgt & " (nth-value 1 (gethash " & Key & " " & Val & ")))");
               when Step_Set_New =>
                  Append_Line ("  (setf " & Tgt & " (make-hash-table :test 'equal))");
               when Step_Set_Add =>
                  Append_Line ("  (setf (gethash " & Val & " " & Tgt & ") t)");
               when Step_Set_Remove =>
                  Append_Line ("  (remhash " & Val & " " & Tgt & ")");
               when Step_Set_Has =>
                  Append_Line ("  (setf " & Tgt & " (nth-value 1 (gethash " & Val & " " & Args & ")))");
               when Step_Struct_New =>
                  Append_Line ("  (setf " & Tgt & " (make-" & Val & "))");
               when Step_Struct_Get =>
                  Append_Line ("  (setf " & Tgt & " (" & Val & "-" & Field & "))");
               when Step_Struct_Set =>
                  Append_Line ("  (setf (" & Tgt & "-" & Field & ") " & Val & ")");
               when Step_Generic_Call =>
                  Append_Line ("  (setf " & Tgt & " (" & Val & " " & Type_Args & " " & Args & "))");
               when Step_Type_Cast =>
                  Append_Line ("  (setf " & Tgt & " (coerce " & Val & " '" & Args & "))");
               when others =>
                  Append_Line ("  ;; unsupported step");
            end case;
         end;
         <<Continue_Common_Lisp>>
      end loop;
   end Emit_Steps_Common_Lisp;

   procedure Emit_Steps_Scheme (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0)  ; no steps");
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
                  Append_Line ("  (do ((i " & Init & " (+ i 1))) ((>= i " & Cond & "))");
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
                                 Append_Line ("    #f");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("  )");
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
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  (error " & Val & ")");
                  else
                     Append_Line ("  (error \"runtime error\")");
                  end if;
               when Step_Break =>
                  Append_Line ("  (break)");
               when Step_Continue =>
                  Append_Line ("  (continue)");
               when Step_Switch =>
                  Append_Line ("  (case " & Val);
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    ((" & Step.Case_Values (C) & ")");
                        for B in Step_Index range Step.Case_Starts (C) .. Step.Case_Starts (C) + Step.Case_Counts (C) - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                                 B_Tgt : constant String := To_String (Body_Step.Target);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Assign =>
                                       if B_Tgt'Length > 0 then
                                          Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
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
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    (t");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("      " & B_Val);
                                 when others =>
                                    Append_Line ("      #f");
                              end case;
                           end;
                        end if;
                     end loop;
                     Append_Line ("    )");
                  end if;
                  Append_Line ("  )");
               when Step_Try =>
                  Append_Line ("  (handler-case");
                  Append_Line ("    (progn");
                  for B in Step_Index range Step.Try_Start .. Step.Try_Start + Step.Try_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      (set! " & B_Tgt & " " & B_Val & ")");
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
                  for C in Step_Index range 1 .. Step.Catch_Count loop
                     if Step.Catch_Blocks (C).Exception_Type'Length > 0 then
                        Append_Line ("    (" & Step.Catch_Blocks (C).Exception_Type & " (err)");
                        Append_Line ("      (progn");
                        for B in Step_Index range Step.Catch_Blocks (C).Handler_Start .. Step.Catch_Blocks (C).Handler_Start + Step.Catch_Blocks (C).Handler_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Return =>
                                       Append_Line ("        " & B_Val);
                                    when others =>
                                       Append_Line ("        #f");
                                 end case;
                              end;
                           end if;
                        end loop;
                        Append_Line ("      ))");
                     end if;
                  end loop;
                  Append_Line ("  )");
               when Step_Throw =>
                  Append_Line ("  (error " & Val & ")");
               when Step_Array_New =>
                  Append_Line ("  (set! " & Tgt & " (make-array " & Val & "))");
               when Step_Array_Get =>
                  Append_Line ("  (set! " & Tgt & " (aref " & Val & " " & Idx & "))");
               when Step_Array_Set =>
                  Append_Line ("  (setf (aref " & Tgt & " " & Idx & ") " & Val & ")");
               when Step_Array_Push =>
                  Append_Line ("  (vector-push-extend " & Val & " " & Tgt & ")");
               when Step_Array_Pop =>
                  Append_Line ("  (set! " & Tgt & " (vector-pop " & Val & "))");
               when Step_Array_Len =>
                  Append_Line ("  (set! " & Tgt & " (length " & Val & "))");
               when Step_Map_New =>
                  Append_Line ("  (set! " & Tgt & " (make-hash-table))");
               when Step_Map_Get =>
                  Append_Line ("  (set! " & Tgt & " (gethash " & Key & " " & Val & "))");
               when Step_Map_Set =>
                  Append_Line ("  (setf (gethash " & Key & " " & Tgt & ") " & Val & ")");
               when Step_Map_Delete =>
                  Append_Line ("  (remhash " & Key & " " & Val & ")");
               when Step_Map_Has =>
                  Append_Line ("  (set! " & Tgt & " (nth-value 1 (gethash " & Key & " " & Val & ")))");
               when Step_Set_New =>
                  Append_Line ("  (set! " & Tgt & " (make-hash-table :test 'equal))");
               when Step_Set_Add =>
                  Append_Line ("  (setf (gethash " & Val & " " & Tgt & ") t)");
               when Step_Set_Remove =>
                  Append_Line ("  (remhash " & Val & " " & Tgt & ")");
               when Step_Set_Has =>
                  Append_Line ("  (set! " & Tgt & " (nth-value 1 (gethash " & Val & " " & Args & ")))");
               when Step_Struct_New =>
                  Append_Line ("  (set! " & Tgt & " (make-" & Val & "))");
               when Step_Struct_Get =>
                  Append_Line ("  (set! " & Tgt & " (" & Val & "-" & Field & "))");
               when Step_Struct_Set =>
                  Append_Line ("  (setf (" & Tgt & "-" & Field & ") " & Val & ")");
               when Step_Generic_Call =>
                  Append_Line ("  (set! " & Tgt & " (" & Val & " " & Type_Args & " " & Args & "))");
               when Step_Type_Cast =>
                  Append_Line ("  (set! " & Tgt & " (coerce " & Val & " '" & Args & "))");
               when others =>
                  Append_Line ("  ; unsupported step");
            end case;
         end;
         <<Continue_Scheme>>
      end loop;
   end Emit_Steps_Scheme;

end Emit_Target.Lisp_Family;