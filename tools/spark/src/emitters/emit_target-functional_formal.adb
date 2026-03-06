--  Emit Target Functional Formal - Functional/Formal Language Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Identifier_Strings;
use Identifier_Strings;

package body Emit_Target.Functional_Formal is

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

   procedure Emit_Steps_Futhark (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Futhark;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
            Init : constant String := To_String (Step.Init);
            Incr : constant String := To_String (Step.Increment);
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
                  declare
                     Then_Assign_Val : Identifier_String := Null_Bounded_String;
                     Else_Assign_Val : Identifier_String := Null_Bounded_String;
                     Assign_Target    : Identifier_String := Null_Bounded_String;
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
                     if Length (Then_Assign_Val) > 0 and Length (Else_Assign_Val) > 0 then
                        Append_Line ("  let " & To_String (Assign_Target) & " = if " & Cond & " then " & To_String (Then_Assign_Val) & " else " & To_String (Else_Assign_Val));
                     else
                        --  Fallback: emit if expression with return values
                        declare
                           Then_Val : constant String := 
                             (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                  else "0")
                              else "0");
                           Else_Val : constant String :=
                             (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
                                  else "0")
                              else "0");
                        begin
                           Append_Line ("  if " & Cond & " then " & Then_Val & " else " & Else_Val);
                        end;
                     end if;
                  end;
               when Step_While =>
                  --  Futhark uses loop expressions
                  Append_Line ("  let " & Tgt & " = loop " & Tgt & " = " & Val & " while " & Cond & " do");
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
                                    Append_Line ("    let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_For =>
                  --  Futhark uses loop expressions with range
                  Append_Line ("  let " & Tgt & " = loop " & Tgt & " = " & Val & " for i < " & Cond & " do");
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
                                    Append_Line ("    let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  let _ = assert false \"" & Val & "\"");
                  else
                     Append_Line ("  let _ = assert false \"error\"");
                  end if;
               when Step_Break =>
                  Append_Line ("  -- break not directly supported");
               when Step_Continue =>
                  Append_Line ("  -- continue not directly supported");
               when Step_Switch =>
                  Append_Line ("  match " & Val & " with");
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    | " & Step.Case_Values (C) & " ->");
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
                                          Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("        " & B_Val);
                                    when others =>
                                       Append_Line ("        0");
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    | _ ->");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("        " & B_Val);
                                 when others =>
                                    Append_Line ("        0");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
               when Step_Try =>
                  Append_Line ("  -- try/except: use Result type");
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
                                    Append_Line ("  let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Throw =>
                  Append_Line ("  Err " & Val);
               when Step_Array_New =>
                  Append_Line ("  let " & Tgt & " = replicate " & Val & " 0");
               when Step_Array_Get =>
                  Append_Line ("  let " & Tgt & " = " & Val & "[" & Idx & "]");
               when Step_Array_Set =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " with [" & Idx & "] = " & Val);
               when Step_Array_Push =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " ++ [" & Val & "]");
               when Step_Array_Pop =>
                  Append_Line ("  let " & Tgt & " = " & Val & "[0:" & Val & ".len-1]");
               when Step_Array_Len =>
                  Append_Line ("  let " & Tgt & " = length " & Val);
               when Step_Map_New =>
                  Append_Line ("  let " & Tgt & " = empty (map i32 i32)");
               when Step_Map_Get =>
                  Append_Line ("  let " & Tgt & " = " & Val & " !" & Key);
               when Step_Map_Set =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " !" & Key & " = " & Val);
               when Step_Set_New =>
                  Append_Line ("  let " & Tgt & " = empty (set i32)");
               when Step_Set_Add =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " `add` " & Val);
               when Step_Set_Remove =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " `delete` " & Val);
               when Step_Set_Has =>
                  Append_Line ("  let " & Tgt & " = " & Args & " `contains` " & Val);
               when Step_Struct_New =>
                  Append_Line ("  let " & Tgt & " = { " & Args & " }");
               when Step_Struct_Get =>
                  Append_Line ("  let " & Tgt & " = " & Val & "." & Field);
               when Step_Struct_Set =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " with ." & Field & " = " & Val);
               when Step_Generic_Call =>
                  Append_Line ("  let " & Tgt & " = " & Val & " " & Type_Args & " " & Args);
               when Step_Type_Cast =>
                  Append_Line ("  let " & Tgt & " = " & Val & " :> " & Args);
               when others =>
                  Append_Line ("  -- unsupported step");
            end case;
         end;
         <<Continue_Futhark>>
      end loop;
   end Emit_Steps_Futhark;

   procedure Emit_Steps_Lean4 (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Lean4;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
            Init : constant String := To_String (Step.Init);
            Incr : constant String := To_String (Step.Increment);
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
                     Then_Assign_Val : Identifier_String := Null_Bounded_String;
                     Else_Assign_Val : Identifier_String := Null_Bounded_String;
                     Assign_Target    : Identifier_String := Null_Bounded_String;
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
                     if Length (Then_Assign_Val) > 0 and Length (Else_Assign_Val) > 0 then
                        Append_Line ("  let " & To_String (Assign_Target) & " := if " & Cond & " then " & To_String (Then_Assign_Val) & " else " & To_String (Else_Assign_Val));
                     else
                        --  Fallback: emit if expression with return values
                        declare
                           Then_Val : constant String := 
                             (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                  else "by admit")
                              else "by admit");
                           Else_Val : constant String :=
                             (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
                                  else "by admit")
                              else "by admit");
                        begin
                           Append_Line ("  if " & Cond & " then " & Then_Val & " else " & Else_Val);
                        end;
                     end if;
                  end;
               when Step_While =>
                  --  Lean4 uses forM or recursion
                  Append_Line ("  let " & Tgt & " ← whileM (pure " & Cond & ") (do");
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
                                    Append_Line ("    let " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("    let " & B_Tgt & " ← " & B_Val & " " & B_Args);
                                 else
                                    Append_Line ("    " & B_Val & " " & B_Args);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    pure ())");
               when Step_For =>
                  --  Lean4 uses forM
                  Append_Line ("  for i in [" & Init & ":" & Cond & "] do");
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
                                    Append_Line ("    let " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("    let " & B_Tgt & " ← " & B_Val & " " & B_Args);
                                 else
                                    Append_Line ("    " & B_Val & " " & B_Args);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
                  --  Lean4 uses forM
                  Append_Line ("  -- for " & Init & "; " & Cond & "; " & Incr);
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
                                    Append_Line ("  --   let " & B_Tgt & " := " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  panic! \"" & Val & "\"");
                  else
                     Append_Line ("  panic! \"error\"");
                  end if;
               when Step_Break =>
                  Append_Line ("  -- break not directly supported");
               when Step_Continue =>
                  Append_Line ("  -- continue not directly supported");
               when Step_Switch =>
                  Append_Line ("  match " & Val & " with");
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    | " & Step.Case_Values (C) & " =>");
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
                                          Append_Line ("        let " & B_Tgt & " := " & B_Val);
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("        " & B_Val);
                                    when others =>
                                       Append_Line ("        by admit");
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    | _ =>");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("        " & B_Val);
                                 when others =>
                                    Append_Line ("        by admit");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
               when Step_Try =>
                  Append_Line ("  -- try/except: use Except monad");
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
                                    Append_Line ("  let " & B_Tgt & " := " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Throw =>
                  Append_Line ("  throw \"" & Val & "\"");
               when Step_Array_New =>
                  Append_Line ("  let " & Tgt & " := List.replicate " & Val & " 0");
               when Step_Array_Get =>
                  Append_Line ("  let " & Tgt & " := " & Val & ".get! " & Idx);
               when Step_Array_Set =>
                  Append_Line ("  let " & Tgt & " := " & Tgt & ".set " & Idx & " " & Val);
               when Step_Array_Push =>
                  Append_Line ("  let " & Tgt & " := " & Tgt & " ++ [" & Val & "]");
               when Step_Array_Pop =>
                  Append_Line ("  let " & Tgt & " := " & Val & ".tail");
               when Step_Array_Len =>
                  Append_Line ("  let " & Tgt & " := " & Val & ".length");
               when Step_Map_New =>
                  Append_Line ("  let " & Tgt & " := HashMap.empty");
               when Step_Map_Get =>
                  Append_Line ("  let " & Tgt & " := " & Val & ".find? " & Key);
               when Step_Map_Set =>
                  Append_Line ("  let " & Tgt & " := " & Tgt & ".insert " & Key & " " & Val);
               when Step_Set_New =>
                  Append_Line ("  let " & Tgt & " := HashSet.empty");
               when Step_Set_Add =>
                  Append_Line ("  let " & Tgt & " := " & Tgt & ".insert " & Val);
               when Step_Set_Remove =>
                  Append_Line ("  let " & Tgt & " := " & Tgt & ".erase " & Val);
               when Step_Set_Has =>
                  Append_Line ("  let " & Tgt & " := " & Args & ".contains " & Val);
               when Step_Struct_New =>
                  Append_Line ("  let " & Tgt & " := { " & Args & " }");
               when Step_Struct_Get =>
                  Append_Line ("  let " & Tgt & " := " & Val & "." & Field);
               when Step_Struct_Set =>
                  Append_Line ("  let " & Tgt & " := { " & Tgt & " with " & Field & " := " & Val & " }");
               when Step_Generic_Call =>
                  Append_Line ("  let " & Tgt & " := @(" & Val & " " & Type_Args & ") " & Args);
               when Step_Type_Cast =>
                  Append_Line ("  let " & Tgt & " := (" & Val & " : " & Args & ")");
               when others =>
                  Append_Line ("  -- unsupported step");
            end case;
         end;
         <<Continue_Lean4>>
      end loop;
   end Emit_Steps_Lean4;

   procedure Emit_Steps_Haskell (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("  0");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Haskell;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := To_String (Step.Value);
            Tgt  : constant String := To_String (Step.Target);
            Cond : constant String := To_String (Step.Condition);
            Args : constant String := To_String (Step.Args);
            Init : constant String := To_String (Step.Init);
            Incr : constant String := To_String (Step.Increment);
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
                     Then_Assign_Val : Identifier_String := Null_Bounded_String;
                     Else_Assign_Val : Identifier_String := Null_Bounded_String;
                     Assign_Target    : Identifier_String := Null_Bounded_String;
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
                     if Length (Then_Assign_Val) > 0 and Length (Else_Assign_Val) > 0 then
                        Append_Line ("  let " & To_String (Assign_Target) & " = if " & Cond & " then " & To_String (Then_Assign_Val) & " else " & To_String (Else_Assign_Val));
                     else
                        --  Fallback: emit if expression with return values
                        declare
                           Then_Val : constant String := 
                             (if Step.Then_Count > 0 and then Step.Then_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Then_Start + Step.Then_Count - 1).Value)
                                  else "()")
                              else "()");
                           Else_Val : constant String :=
                             (if Step.Else_Count > 0 and then Step.Else_Start <= Func.Steps.Count then
                                 (if Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Step_Type = Step_Return then
                                     To_String (Func.Steps.Steps (Step.Else_Start + Step.Else_Count - 1).Value)
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
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
                           B_Val : constant String := To_String (Body_Step.Value);
                           B_Tgt : constant String := To_String (Body_Step.Target);
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
               when Step_Error =>
                  if Val'Length > 0 then
                     Append_Line ("  error \"" & Val & "\"");
                  else
                     Append_Line ("  error \"runtime error\"");
                  end if;
               when Step_Break =>
                  Append_Line ("  -- break not directly supported");
               when Step_Continue =>
                  Append_Line ("  -- continue not directly supported");
               when Step_Switch =>
                  Append_Line ("  case " & Val & " of");
                  for C in Case_Index range 1 .. Step.Case_Count loop
                     if Step.Case_Starts (C) <= Func.Steps.Count then
                        Append_Line ("    " & Step.Case_Values (C) & " ->");
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
                                          Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                       end if;
                                    when Step_Return =>
                                       Append_Line ("        " & B_Val);
                                    when others =>
                                       Append_Line ("        ()");
                                 end case;
                              end;
                           end if;
                        end loop;
                     end if;
                  end loop;
                  if Step.Default_Count > 0 then
                     Append_Line ("    _ ->");
                     for B in Step_Index range Step.Default_Start .. Step.Default_Start + Step.Default_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := To_String (Body_Step.Value);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Return =>
                                    Append_Line ("        " & B_Val);
                                 when others =>
                                    Append_Line ("        ()");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
               when Step_Try =>
                  Append_Line ("  -- try/except: use Either or ExceptT");
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
                                    Append_Line ("  let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when others =>
                                 null;
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Throw =>
                  Append_Line ("  Left \"" & Val & "\"");
               when Step_Array_New =>
                  Append_Line ("  let " & Tgt & " = replicate " & Val & " 0");
               when Step_Array_Get =>
                  Append_Line ("  let " & Tgt & " = " & Val & " !! " & Idx);
               when Step_Array_Set =>
                  Append_Line ("  let " & Tgt & " = take " & Idx & " " & Tgt & " ++ [" & Val & "] ++ drop (" & Idx & " + 1) " & Tgt);
               when Step_Array_Push =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " ++ [" & Val & "]");
               when Step_Array_Pop =>
                  Append_Line ("  let " & Tgt & " = init " & Val);
               when Step_Array_Len =>
                  Append_Line ("  let " & Tgt & " = length " & Val);
               when Step_Map_New =>
                  Append_Line ("  let " & Tgt & " = Map.empty");
               when Step_Map_Get =>
                  Append_Line ("  let " & Tgt & " = Map.lookup " & Key & " " & Val);
               when Step_Map_Set =>
                  Append_Line ("  let " & Tgt & " = Map.insert " & Key & " " & Val & " " & Tgt);
               when Step_Set_New =>
                  Append_Line ("  let " & Tgt & " = Set.empty");
               when Step_Set_Add =>
                  Append_Line ("  let " & Tgt & " = Set.insert " & Val & " " & Tgt);
               when Step_Set_Remove =>
                  Append_Line ("  let " & Tgt & " = Set.delete " & Val & " " & Tgt);
               when Step_Set_Has =>
                  Append_Line ("  let " & Tgt & " = Set.member " & Val & " " & Args);
               when Step_Struct_New =>
                  Append_Line ("  let " & Tgt & " = " & Val & " { " & Args & " }");
               when Step_Struct_Get =>
                  Append_Line ("  let " & Tgt & " = " & Val & "." & Field);
               when Step_Struct_Set =>
                  Append_Line ("  let " & Tgt & " = " & Tgt & " { " & Field & " = " & Val & " }");
               when Step_Generic_Call =>
                  Append_Line ("  let " & Tgt & " = " & Val & " @" & Type_Args & " " & Args);
               when Step_Type_Cast =>
                  Append_Line ("  let " & Tgt & " = " & Val & " :: " & Args);
               when others =>
                  Append_Line ("  -- unsupported step");
            end case;
         end;
         <<Continue_Haskell>>
      end loop;
   end Emit_Steps_Haskell;

end Emit_Target.Functional_Formal;