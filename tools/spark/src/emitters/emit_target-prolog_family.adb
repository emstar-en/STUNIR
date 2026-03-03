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
                  Append_Line ("    % while loop: use recursion");
                  Append_Line ("    % condition: " & Cond);
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
                  Append_Line ("    % for loop: use for/3 or between/3");
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
                  Append_Line ("    % while loop: use recursion with accumulator");
                  Append_Line ("    % condition: " & Cond);
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
                                    Append_Line ("    %   " & B_Tgt & " = " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_For =>
                  Append_Line ("    % for loop: use fold/4 or recursion");
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
                                    Append_Line ("    %   " & B_Tgt & " = " & B_Val);
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
         <<Continue_Mercury>>
      end loop;
   end Emit_Steps_Mercury;

end Emit_Target.Prolog_Family;