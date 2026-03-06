--  Emit Target Prolog Family - Prolog Dialect Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Identifier_Strings;
use Identifier_Strings;

package body Emit_Target.Prolog_Family is

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

   procedure Emit_Steps_SWI_Prolog (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    % no steps");
         Append_Line ("    Result = nil.");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_SWI;
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                  Append_Line ("    % pattern: while_loop(State, Result) :- " & Cond & ", !, body_step(State, NewState), while_loop(NewState, Result).");
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
                                    Append_Line ("    %   " & B_Tgt & " is " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_For =>
                  Append_Line ("    % for loop: use between/3 or recursion");
                  Append_Line ("    % pattern: between(Low, High, I), body_step(I, Result).");
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
                                    Append_Line ("    %   " & B_Tgt & " is " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Switch =>
                  Append_Line ("    % switch: use multiple clauses");
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    % case " & To_String (Step.Case_Values (C)) & ":");
                        for B in Step_Index range Step.Case_Starts (C) .. Step.Case_Starts (C) + Step.Case_Counts (C) - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Return =>
                                       Append_Line ("    %   Result = " & B_Val);
                                    when others =>
                                       null;
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                  end loop;
               when Step_Try =>
                  Append_Line ("    % try/catch: use catch/3");
                  Append_Line ("    catch(");
                  for B in Step_Index range Step.Try_Start .. Step.Try_Start + Step.Try_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := To_String (Body_Step.Value);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Call =>
                                 Append_Line ("        " & B_Val & "(_)");
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  for C in Step_Index range 1 .. Step.Catch_Count loop
                     if Step.Catch_Blocks (C).Exception_Type'Length > 0 then
                        Append_Line ("    ), " & To_String (Step.Catch_Blocks (C).Exception_Type) & ",");
                        Append_Line ("    % handler:");
                        for B in Step_Index range Step.Catch_Blocks (C).Handler_Start .. Step.Catch_Blocks (C).Handler_Start + Step.Catch_Blocks (C).Handler_Count - 1 loop
                           if B <= Func.Steps.Count then
                              declare
                                 Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                 B_Val : constant String := To_String (Body_Step.Value);
                              begin
                                 case Body_Step.Step_Type is
                                    when Step_Return =>
                                       Append_Line ("    %   Result = " & B_Val);
                                    when others =>
                                       null;
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                  end loop;
                  Append_Line ("    ),");
               when Step_Throw =>
                  Append_Line ("    throw(" & Val & "),");
               when Step_Array_New =>
                  Append_Line ("    " & Tgt & " = [],"),
               when Step_Array_Get =>
                  Append_Line ("    nth0(" & Args & ", " & Val & ", " & Tgt & "),"),
               when Step_Array_Set =>
                  Append_Line ("    setarg(" & Args & ", " & Tgt & ", " & Val & "),"),
               when Step_Array_Push =>
                  Append_Line ("    append(" & Tgt & ", [" & Val & "], " & Tgt & "),"),
               when Step_Array_Pop =>
                  Append_Line ("    append(" & Tgt & ", [_], " & Tgt & "),"),
               when Step_Array_Len =>
                  Append_Line ("    length(" & Val & ", " & Tgt & "),"),
               when Step_Map_New =>
                  Append_Line ("    " & Tgt & " = {},"),
               when Step_Map_Get =>
                  Append_Line ("    get_dict(" & Args & ", " & Val & ", " & Tgt & "),"),
               when Step_Map_Set =>
                  Append_Line ("    put_dict(" & Args & ", " & Tgt & ", " & Val & "),"),
               when Step_Set_New =>
                  Append_Line ("    list_to_set([], " & Tgt & "),"),
               when Step_Set_Add =>
                  Append_Line ("    list_to_set([" & Val & "|_], " & Tgt & "),"),
               when Step_Set_Has =>
                  Append_Line ("    member(" & Args & ", " & Val & "),"),
               when others =>
                  Append_Line ("    % unsupported step");
            end case;
         end;
         <<Continue_SWI>>
      end loop;
   end Emit_Steps_SWI_Prolog;

   procedure Emit_Steps_GNU_Prolog (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      --  GNU Prolog is very similar to SWI-Prolog
      --  Main differences: some built-in predicates, FD solver syntax
      if Func.Steps.Count = 0 then
         Append_Line ("    % no steps");
         Append_Line ("    Result = nil.");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_GNU;
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
                           B_Args : constant String := To_String (Body_Step.Args);
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
                              B_Val : constant String := To_String (Body_Step.Value);
                              B_Tgt : constant String := To_String (Body_Step.Target);
                              B_Args : constant String := To_String (Body_Step.Args);
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
                  --  SWI-Prolog while loop using recursion
                  Append_Line ("    (   " & Cond);
                  Append_Line ("    ->  (");
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
                                    Append_Line ("            " & B_Tgt & " is " & B_Val & ",");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Val & "(" & B_Args & ", " & B_Tgt & "),");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & "),");
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("            fail");
                  Append_Line ("        ;   true");
                  Append_Line ("        )");
                  Append_Line ("    ;   true");
                  Append_Line ("    ),");
               when Step_For =>
                  --  SWI-Prolog for loop using between/3
                  Append_Line ("    between(" & Init & ", " & Cond & ", _I),");
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
                                    Append_Line ("    " & B_Tgt & " is " & B_Val & ",");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("    " & B_Val & "(" & B_Args & ", " & B_Tgt & "),");
                                 else
                                    Append_Line ("    " & B_Val & "(" & B_Args & "),");
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when others =>
                  Append_Line ("    % unsupported step");
            end case;
         end;
         <<Continue_GNU>>
      end loop;
   end Emit_Steps_GNU_Prolog;

   procedure Emit_Steps_Mercury (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      --  Mercury is a functional/logic language with strong typing
      --  Different syntax from Prolog: uses mode declarations, determinism
      if Func.Steps.Count = 0 then
         Append_Line ("    % no steps");
         Append_Line ("    Result = nil.");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Mercury;
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
                     Append_Line ("    " & Tgt & " = " & Val & ",");
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val & "(" & Args & "),");
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
                  Append_Line ("    ( if " & Cond & " then");
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
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ",");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & "),");
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
                     Append_Line ("    else");
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
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & ",");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & "),");
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
                  --  Mercury while loop using recursion with accumulator
                  Append_Line ("    ( if " & Cond & " then");
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
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ",");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & "),");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & "),");
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        % recurse with updated state");
                  Append_Line ("    else");
                  Append_Line ("        true");
                  Append_Line ("    ),");
               when Step_For =>
                  --  Mercury for loop using fold/4 or recursion
                  Append_Line ("    ( for _I in " & Init & " .. " & Cond & " do");
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
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & ",");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & "),");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & "),");
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    ),");
               when others =>
                  Append_Line ("    % unsupported step");
            end case;
         end;
         <<Continue_Mercury>>
      end loop;
   end Emit_Steps_Mercury;

end Emit_Target.Prolog_Family;