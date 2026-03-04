--  Emit Target Mainstream - Mainstream Language Code Emission
--  Phase: 3 (Emit)
--  SPARK_Mode: On
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with STUNIR_Types;
use STUNIR_Types;

package body Emit_Target.Mainstream is

   --  Use clauses for nested packages
   package IStr renames Identifier_Strings;
   package TStr renames Type_Name_Strings;

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

   procedure Emit_Steps_Rust (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    todo!()");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Rust;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("    " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    " & Val);
                  else
                     Append_Line ("    ()");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    while " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i in " & Init & ".." & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Break =>
                  Append_Line ("    break;");
               when Step_Continue =>
                  Append_Line ("    continue;");
               when Step_Error =>
                  Append_Line ("    return Err(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("    match " & Cond & " {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("        " & Case_Val & " => {");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            let " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("            " & B_Val);
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("        }");
                        end;
                     end if;
                  end loop;
                  Append_Line ("        _ => {}");
                  Append_Line ("    }");
               when Step_Try =>
                  Append_Line ("    let " & Tgt & " = match " & Val & " {");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Type : constant String := IStr.To_String (Step.Catch_Blocks (C).Exception_Type);
                           Catch_Var  : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("        Err(" & Catch_Var & ": " & Catch_Type & ") => {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            let " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("        }");
                        end;
                     end if;
                  end loop;
                  Append_Line ("        Ok(v) => v,");
                  Append_Line ("    };");
               when Step_Throw =>
                  Append_Line ("    return Err(""" & Val & """.into());");
               when Step_Array_New =>
                  Append_Line ("    let " & Tgt & ": Vec<" & Args & "> = vec![];");
               when Step_Array_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & ".push(" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".pop();");
               when Step_Array_Len =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".len();");
               when Step_Map_New =>
                  Append_Line ("    let " & Tgt & ": HashMap<" & Args & "> = HashMap::new();");
               when Step_Map_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".get(&" & Args & ");");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & ".insert(" & Args & ", " & Val & ");");
               when Step_Map_Delete =>
                  Append_Line ("    " & Tgt & ".remove(&" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".contains_key(&" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("    let " & Tgt & ": Vec<_> = " & Val & ".keys().collect();");
               when Step_Set_New =>
                  Append_Line ("    let " & Tgt & ": HashSet<" & Args & "> = HashSet::new();");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & ".insert(" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("    " & Tgt & ".remove(&" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".contains(&" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("    let " & Tgt & ": HashSet<_> = " & Val & ".union(&" & Args & ").collect();");
               when Step_Set_Intersect =>
                  Append_Line ("    let " & Tgt & ": HashSet<_> = " & Val & ".intersection(&" & Args & ").collect();");
               when Step_Struct_New =>
                  Append_Line ("    let " & Tgt & " = " & Args & " { " & Val & " };");
               when Step_Struct_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("    let " & Tgt & " = " & Val & "::<" & Args & ">();");
               when Step_Type_Cast =>
                  Append_Line ("    let " & Tgt & " = " & Val & " as " & Args & ";");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    // unsupported step");
            end case;
         end;
         <<Continue_Rust>>
      end loop;
   end Emit_Steps_Rust;

   procedure Emit_Steps_Python (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    return None");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Python;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val);
                  else
                     Append_Line ("    # assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return None");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & ":");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("    else:");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        pass");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
               when Step_While =>
                  Append_Line ("    while " & Cond & ":");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_For =>
                  Append_Line ("    for i in range(" & Init & ", " & Cond & "):");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        pass");
                           end case;
                        end;
                     end if;
                  end loop;
               when Step_Break =>
                  Append_Line ("    break");
               when Step_Continue =>
                  Append_Line ("    continue");
               when Step_Error =>
                  Append_Line ("    raise RuntimeError(""" & Val & """)");
               when Step_Switch =>
                  Append_Line ("    match " & Cond & ":");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("        case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            " & B_Tgt & " = " & B_Val);
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("            return " & B_Val);
                                       when others =>
                                          Append_Line ("            pass");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        case _:");
                  Append_Line ("            pass");
               when Step_Try =>
                  Append_Line ("    try:");
                  Append_Line ("        " & Tgt & " = " & Val);
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Type : constant String := IStr.To_String (Step.Catch_Blocks (C).Exception_Type);
                           Catch_Var  : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("    except " & Catch_Type & " as " & Catch_Var & ":");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        " & B_Tgt & " = " & B_Val);
                                          end if;
                                       when others =>
                                          Append_Line ("        pass");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
               when Step_Throw =>
                  Append_Line ("    raise Exception(""" & Val & """)");
               when Step_Array_New =>
                  Append_Line ("    " & Tgt & " = []");
               when Step_Array_Get =>
                  Append_Line ("    " & Tgt & " = " & Val & "[" & Args & "]");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val);
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & ".append(" & Val & ")");
               when Step_Array_Pop =>
                  Append_Line ("    " & Tgt & " = " & Val & ".pop()");
               when Step_Array_Len =>
                  Append_Line ("    " & Tgt & " = len(" & Val & ")");
               when Step_Map_New =>
                  Append_Line ("    " & Tgt & " = {}");
               when Step_Map_Get =>
                  Append_Line ("    " & Tgt & " = " & Val & ".get(" & Args & ")");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val);
               when Step_Map_Delete =>
                  Append_Line ("    del " & Tgt & "[" & Val & "]");
               when Step_Map_Has =>
                  Append_Line ("    " & Tgt & " = " & Args & " in " & Val);
               when Step_Map_Keys =>
                  Append_Line ("    " & Tgt & " = list(" & Val & ".keys())");
               when Step_Set_New =>
                  Append_Line ("    " & Tgt & " = set()");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & ".add(" & Val & ")");
               when Step_Set_Remove =>
                  Append_Line ("    " & Tgt & ".discard(" & Val & ")");
               when Step_Set_Has =>
                  Append_Line ("    " & Tgt & " = " & Args & " in " & Val);
               when Step_Set_Union =>
                  Append_Line ("    " & Tgt & " = " & Val & " | " & Args);
               when Step_Set_Intersect =>
                  Append_Line ("    " & Tgt & " = " & Val & " & " & Args);
               when Step_Struct_New =>
                  Append_Line ("    " & Tgt & " = {" & Val & "}");
               when Step_Struct_Get =>
                  Append_Line ("    " & Tgt & " = " & Val & "." & Args);
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val);
               when Step_Generic_Call =>
                  Append_Line ("    " & Tgt & " = " & Val & "(" & Args & ")");
               when Step_Type_Cast =>
                  Append_Line ("    " & Tgt & " = " & Args & "(" & Val & ")");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    # unsupported step");
            end case;
         end;
         <<Continue_Python>>
      end loop;
   end Emit_Steps_Python;

   procedure Emit_Steps_Go (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    return");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Go;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " := " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    " & Tgt & " := " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " := " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    for " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i := " & Init & "; i < " & Cond & "; i++ {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        " & B_Tgt & " := " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Break =>
                  Append_Line ("    break;");
               when Step_Continue =>
                  Append_Line ("    continue;");
               when Step_Error =>
                  Append_Line ("    return errors.New(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("    switch " & Cond & " {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("    case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("        return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("        // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    default:");
                  Append_Line ("        // default case");
                  Append_Line ("    }");
               when Step_Try =>
                  Append_Line ("    defer func() {");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Var : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("        if r := recover(); r != nil {");
                           Append_Line ("            " & Catch_Var & " := r");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("        }");
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }();");
                  Append_Line ("    " & Tgt & " := " & Val & ";");
               when Step_Throw =>
                  Append_Line ("    panic(""" & Val & """);");
               when Step_Array_New =>
                  Append_Line ("    " & Tgt & " := make([]" & Args & ", 0);");
               when Step_Array_Get =>
                  Append_Line ("    " & Tgt & " := " & Val & "[" & Args & "];");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & " = append(" & Tgt & ", " & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("    " & Tgt & " := " & Val & "[len(" & Val & ")-1];");
                  Append_Line ("    " & Val & " = " & Val & "[:len(" & Val & ")-1];");
               when Step_Array_Len =>
                  Append_Line ("    " & Tgt & " := len(" & Val & ");");
               when Step_Map_New =>
                  Append_Line ("    " & Tgt & " := make(map[" & Args & "]);");
               when Step_Map_Get =>
                  Append_Line ("    " & Tgt & " := " & Val & "[" & Args & "];");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Map_Delete =>
                  Append_Line ("    delete(" & Tgt & ", " & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("    _, " & Tgt & " := " & Val & "[" & Args & "];");
               when Step_Map_Keys =>
                  Append_Line ("    " & Tgt & " := make([]string, 0);");
                  Append_Line ("    for k := range " & Val & " { " & Tgt & " = append(" & Tgt & ", k) };");
               when Step_Set_New =>
                  Append_Line ("    " & Tgt & " := make(map[" & Args & "]struct{});");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & "[" & Val & "] = struct{}{};");
               when Step_Set_Remove =>
                  Append_Line ("    delete(" & Tgt & ", " & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("    _, " & Tgt & " := " & Val & "[" & Args & "];");
               when Step_Set_Union =>
                  Append_Line ("    // set union: " & Tgt & " = " & Val & " | " & Args);
               when Step_Set_Intersect =>
                  Append_Line ("    // set intersect: " & Tgt & " = " & Val & " & " & Args);
               when Step_Struct_New =>
                  Append_Line ("    " & Tgt & " := " & Args & "{" & Val & "};");
               when Step_Struct_Get =>
                  Append_Line ("    " & Tgt & " := " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("    " & Tgt & " := " & Val & "(" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("    " & Tgt & " := " & Args & "(" & Val & ");");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    // unsupported step");
            end case;
         end;
         <<Continue_Go>>
      end loop;
   end Emit_Steps_Go;

   procedure Emit_Steps_Java (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("        return;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Java;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("        // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("        " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("        return " & Val & ";");
                  else
                     Append_Line ("        return;");
                  end if;
               when Step_If =>
                  Append_Line ("        if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("            return " & B_Val & ";");
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("        } else {");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("            return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("            // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("        }");
               when Step_While =>
                  Append_Line ("        while (" & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_For =>
                  Append_Line ("        for (int i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_Break =>
                  Append_Line ("        break;");
               when Step_Continue =>
                  Append_Line ("        continue;");
               when Step_Error =>
                  Append_Line ("        throw new RuntimeException(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("        switch (" & Cond & ") {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("            case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("                " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("                return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("                // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("                break;");
                        end;
                     end if;
                  end loop;
                  Append_Line ("            default:");
                  Append_Line ("                break;");
                  Append_Line ("        }");
               when Step_Try =>
                  Append_Line ("        try {");
                  Append_Line ("            " & Tgt & " = " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Type : constant String := IStr.To_String (Step.Catch_Blocks (C).Exception_Type);
                           Catch_Var  : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("        } catch (" & Catch_Type & " " & Catch_Var & ") {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_Throw =>
                  Append_Line ("        throw new Exception(""" & Val & """);");
               when Step_Array_New =>
                  Append_Line ("        " & Tgt & " = new ArrayList<" & Args & ">();");
               when Step_Array_Get =>
                  Append_Line ("        " & Tgt & " = " & Val & ".get(" & Args & ");");
               when Step_Array_Set =>
                  Append_Line ("        " & Tgt & ".set(" & Args & ", " & Val & ");");
               when Step_Array_Push =>
                  Append_Line ("        " & Tgt & ".add(" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("        " & Tgt & " = " & Val & ".remove(" & Val & ".size() - 1);");
               when Step_Array_Len =>
                  Append_Line ("        " & Tgt & " = " & Val & ".size();");
               when Step_Map_New =>
                  Append_Line ("        " & Tgt & " = new HashMap<" & Args & ">();");
               when Step_Map_Get =>
                  Append_Line ("        " & Tgt & " = " & Val & ".get(" & Args & ");");
               when Step_Map_Set =>
                  Append_Line ("        " & Tgt & ".put(" & Args & ", " & Val & ");");
               when Step_Map_Delete =>
                  Append_Line ("        " & Tgt & ".remove(" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("        " & Tgt & " = " & Val & ".containsKey(" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("        " & Tgt & " = new ArrayList<>(" & Val & ".keySet());");
               when Step_Set_New =>
                  Append_Line ("        " & Tgt & " = new HashSet<" & Args & ">();");
               when Step_Set_Add =>
                  Append_Line ("        " & Tgt & ".add(" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("        " & Tgt & ".remove(" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("        " & Tgt & " = " & Val & ".contains(" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("        " & Tgt & " = new HashSet<>(" & Val & ");");
                  Append_Line ("        " & Tgt & ".addAll(" & Args & ");");
               when Step_Set_Intersect =>
                  Append_Line ("        " & Tgt & " = new HashSet<>(" & Val & ");");
                  Append_Line ("        " & Tgt & ".retainAll(" & Args & ");");
               when Step_Struct_New =>
                  Append_Line ("        " & Tgt & " = new " & Args & "(" & Val & ");");
               when Step_Struct_Get =>
                  Append_Line ("        " & Tgt & " = " & Val & ".get" & Args & "();");
               when Step_Struct_Set =>
                  Append_Line ("        " & Tgt & ".set" & Args & "(" & Val & ");");
               when Step_Generic_Call =>
                  Append_Line ("        " & Tgt & " = " & Val & "(" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("        " & Tgt & " = (" & Args & ") " & Val & ";");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("        // unsupported step");
            end case;
         end;
         <<Continue_Java>>
      end loop;
   end Emit_Steps_Java;

   procedure Emit_Steps_JavaScript (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    return");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_JS;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
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
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val & ";");
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val & ";");
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
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for (let i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Break =>
                  Append_Line ("    break;");
               when Step_Continue =>
                  Append_Line ("    continue;");
               when Step_Error =>
                  Append_Line ("    throw new Error(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("    switch (" & Cond & ") {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("        case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("            return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("            break;");
                        end;
                     end if;
                  end loop;
                  Append_Line ("        default:");
                  Append_Line ("            break;");
                  Append_Line ("    }");
               when Step_Try =>
                  Append_Line ("    try {");
                  Append_Line ("        " & Tgt & " = " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Var : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("    } catch (" & Catch_Var & ") {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("        // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Throw =>
                  Append_Line ("    throw new Error(""" & Val & """);");
               when Step_Array_New =>
                  Append_Line ("    let " & Tgt & " = [];");
               when Step_Array_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & ".push(" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".pop();");
               when Step_Array_Len =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".length;");
               when Step_Map_New =>
                  Append_Line ("    let " & Tgt & " = new Map();");
               when Step_Map_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".get(" & Args & ");");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & ".set(" & Args & ", " & Val & ");");
               when Step_Map_Delete =>
                  Append_Line ("    " & Tgt & ".delete(" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".has(" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("    let " & Tgt & " = Array.from(" & Val & ".keys());");
               when Step_Set_New =>
                  Append_Line ("    let " & Tgt & " = new Set();");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & ".add(" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("    " & Tgt & ".delete(" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".has(" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("    let " & Tgt & " = new Set([..." & Val & ", ..." & Args & "]);");
               when Step_Set_Intersect =>
                  Append_Line ("    let " & Tgt & " = new Set([..." & Val & "].filter(x => " & Args & ".has(x)));");
               when Step_Struct_New =>
                  Append_Line ("    let " & Tgt & " = { " & Val & " };");
               when Step_Struct_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("    let " & Tgt & " = " & Val & " as " & Args & ";");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    // unsupported step");
            end case;
         end;
         <<Continue_JS>>
      end loop;
   end Emit_Steps_JavaScript;

   procedure Emit_Steps_CSharp (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("        return;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_CSharp;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & ";");
                  else
                     Append_Line ("        // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("        " & Tgt & " = " & Val & "(" & Args & ");");
                  else
                     Append_Line ("        " & Val & "(" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("        return " & Val & ";");
                  else
                     Append_Line ("        return;");
                  end if;
               when Step_If =>
                  Append_Line ("        if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("            return " & B_Val & ";");
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("        } else {");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                    else
                                       Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("            return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("            // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("        }");
               when Step_While =>
                  Append_Line ("        while (" & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_For =>
                  Append_Line ("        for (int i = " & Init & "; i < " & Cond & "; i++) {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("            " & B_Tgt & " = " & B_Val & "(" & B_Args & ");");
                                 else
                                    Append_Line ("            " & B_Val & "(" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("            // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_Break =>
                  Append_Line ("        break;");
               when Step_Continue =>
                  Append_Line ("        continue;");
               when Step_Error =>
                  Append_Line ("        throw new Exception(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("        switch (" & Cond & ") {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("            case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("                " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("                return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("                // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("                break;");
                        end;
                     end if;
                  end loop;
                  Append_Line ("            default:");
                  Append_Line ("                break;");
                  Append_Line ("        }");
               when Step_Try =>
                  Append_Line ("        try {");
                  Append_Line ("            " & Tgt & " = " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Type : constant String := IStr.To_String (Step.Catch_Blocks (C).Exception_Type);
                           Catch_Var  : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("        } catch (" & Catch_Type & " " & Catch_Var & ") {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("        }");
               when Step_Throw =>
                  Append_Line ("        throw new Exception(""" & Val & """);");
               when Step_Array_New =>
                  Append_Line ("        var " & Tgt & " = new List<" & Args & ">();");
               when Step_Array_Get =>
                  Append_Line ("        var " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Array_Set =>
                  Append_Line ("        " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Array_Push =>
                  Append_Line ("        " & Tgt & ".Add(" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("        var " & Tgt & " = " & Val & "[" & Val & ".Count - 1];");
                  Append_Line ("        " & Val & ".RemoveAt(" & Val & ".Count - 1);");
               when Step_Array_Len =>
                  Append_Line ("        var " & Tgt & " = " & Val & ".Count;");
               when Step_Map_New =>
                  Append_Line ("        var " & Tgt & " = new Dictionary<" & Args & ">();");
               when Step_Map_Get =>
                  Append_Line ("        var " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Map_Set =>
                  Append_Line ("        " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Map_Delete =>
                  Append_Line ("        " & Tgt & ".Remove(" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("        var " & Tgt & " = " & Val & ".ContainsKey(" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("        var " & Tgt & " = new List<" & Args & ">(" & Val & ".Keys);");
               when Step_Set_New =>
                  Append_Line ("        var " & Tgt & " = new HashSet<" & Args & ">();");
               when Step_Set_Add =>
                  Append_Line ("        " & Tgt & ".Add(" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("        " & Tgt & ".Remove(" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("        var " & Tgt & " = " & Val & ".Contains(" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("        var " & Tgt & " = new HashSet<" & Args & ">(" & Val & ");");
                  Append_Line ("        " & Tgt & ".UnionWith(" & Args & ");");
               when Step_Set_Intersect =>
                  Append_Line ("        var " & Tgt & " = new HashSet<" & Args & ">(" & Val & ");");
                  Append_Line ("        " & Tgt & ".IntersectWith(" & Args & ");");
               when Step_Struct_New =>
                  Append_Line ("        var " & Tgt & " = new " & Args & " { " & Val & " };");
               when Step_Struct_Get =>
                  Append_Line ("        var " & Tgt & " = " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("        " & Tgt & "." & Args & " = " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("        var " & Tgt & " = " & Val & "<" & Args & ">();");
               when Step_Type_Cast =>
                  Append_Line ("        var " & Tgt & " = (" & Args & ")" & Val & ";");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("        // unsupported step");
            end case;
         end;
         <<Continue_CSharp>>
      end loop;
   end Emit_Steps_CSharp;

   procedure Emit_Steps_Swift (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    return");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Swift;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if " & Cond & " {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
                                 when others =>
                                    Append_Line ("        // step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("    }");
               when Step_While =>
                  Append_Line ("    while " & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for i in " & Init & "..<" & Cond & " {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        let " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Break =>
                  Append_Line ("    break;");
               when Step_Continue =>
                  Append_Line ("    continue;");
               when Step_Error =>
                  Append_Line ("    fatalError(""" & Val & """);");
               when Step_Switch =>
                  Append_Line ("    switch " & Cond & " {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("    case " & Case_Val & ":");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("        return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("        // step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    default:");
                  Append_Line ("        break;");
                  Append_Line ("    }");
               when Step_Try =>
                  Append_Line ("    do {");
                  Append_Line ("        let " & Tgt & " = try " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Var : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("    } catch let " & Catch_Var & " {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        let " & B_Tgt & " = " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("        // step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Throw =>
                  Append_Line ("    throw """ & Val & """;");
               when Step_Array_New =>
                  Append_Line ("    let " & Tgt & ": [" & Args & "] = [];");
               when Step_Array_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & ".append(" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".popLast();");
               when Step_Array_Len =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".count;");
               when Step_Map_New =>
                  Append_Line ("    let " & Tgt & ": [" & Args & "] = [:];");
               when Step_Map_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "[" & Args & "];");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val & ";");
               when Step_Map_Delete =>
                  Append_Line ("    " & Tgt & ".removeValue(forKey: " & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".contains(key: " & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("    let " & Tgt & " = Array(" & Val & ".keys);");
               when Step_Set_New =>
                  Append_Line ("    let " & Tgt & ": Set<" & Args & "> = [];");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & ".insert(" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("    " & Tgt & ".remove(" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".contains(" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".union(" & Args & ");");
               when Step_Set_Intersect =>
                  Append_Line ("    let " & Tgt & " = " & Val & ".intersection(" & Args & ");");
               when Step_Struct_New =>
                  Append_Line ("    let " & Tgt & " = " & Args & "(" & Val & ");");
               when Step_Struct_Get =>
                  Append_Line ("    let " & Tgt & " = " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("    let " & Tgt & " = " & Val & "(" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("    let " & Tgt & " = " & Val & " as! " & Args & ";");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    // unsupported step");
            end case;
         end;
         <<Continue_Swift>>
      end loop;
   end Emit_Steps_Swift;

   procedure Emit_Steps_Kotlin (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("    return");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Kotlin;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("    val " & Tgt & " = " & Val);
                  else
                     Append_Line ("    // assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("    val " & Tgt & " = " & Val & "(" & Args & ")");
                  else
                     Append_Line ("    " & Val & "(" & Args & ")");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("    return " & Val);
                  else
                     Append_Line ("    return");
                  end if;
               when Step_If =>
                  Append_Line ("    if (" & Cond & ") {");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("        return " & B_Val);
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
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                    else
                                       Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("        return " & B_Val);
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
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_For =>
                  Append_Line ("    for (i in " & Init & " until " & Cond & ") {");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("        val " & B_Tgt & " = " & B_Val & "(" & B_Args & ")");
                                 else
                                    Append_Line ("        " & B_Val & "(" & B_Args & ")");
                                 end if;
                              when others =>
                                 Append_Line ("        // step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Break =>
                  Append_Line ("    break");
               when Step_Continue =>
                  Append_Line ("    continue");
               when Step_Error =>
                  Append_Line ("    throw IllegalArgumentException(""" & Val & """)");
               when Step_Switch =>
                  Append_Line ("    when (" & Cond & ") {");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("        " & Case_Val & " -> {");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("            val " & B_Tgt & " = " & B_Val);
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("            return " & B_Val);
                                       when others =>
                                          Append_Line ("            // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                           Append_Line ("        }");
                        end;
                     end if;
                  end loop;
                  Append_Line ("        else -> { }");
                  Append_Line ("    }");
               when Step_Try =>
                  Append_Line ("    try {");
                  Append_Line ("        val " & Tgt & " = " & Val);
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Type : constant String := IStr.To_String (Step.Catch_Blocks (C).Exception_Type);
                           Catch_Var  : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("    } catch (" & Catch_Var & ": " & Catch_Type & ") {");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("        val " & B_Tgt & " = " & B_Val);
                                          end if;
                                       when others =>
                                          Append_Line ("        // step");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("    }");
               when Step_Throw =>
                  Append_Line ("    throw Exception(""" & Val & """)");
               when Step_Array_New =>
                  Append_Line ("    val " & Tgt & " = mutableListOf<" & Args & ">()");
               when Step_Array_Get =>
                  Append_Line ("    val " & Tgt & " = " & Val & "[" & Args & "]");
               when Step_Array_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val);
               when Step_Array_Push =>
                  Append_Line ("    " & Tgt & ".add(" & Val & ")");
               when Step_Array_Pop =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".removeLast()");
               when Step_Array_Len =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".size");
               when Step_Map_New =>
                  Append_Line ("    val " & Tgt & " = mutableMapOf<" & Args & ">()");
               when Step_Map_Get =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".get(" & Args & ")");
               when Step_Map_Set =>
                  Append_Line ("    " & Tgt & "[" & Args & "] = " & Val);
               when Step_Map_Delete =>
                  Append_Line ("    " & Tgt & ".remove(" & Val & ")");
               when Step_Map_Has =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".containsKey(" & Args & ")");
               when Step_Map_Keys =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".keys.toList()");
               when Step_Set_New =>
                  Append_Line ("    val " & Tgt & " = mutableSetOf<" & Args & ">()");
               when Step_Set_Add =>
                  Append_Line ("    " & Tgt & ".add(" & Val & ")");
               when Step_Set_Remove =>
                  Append_Line ("    " & Tgt & ".remove(" & Val & ")");
               when Step_Set_Has =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".contains(" & Args & ")");
               when Step_Set_Union =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".union(" & Args & ")");
               when Step_Set_Intersect =>
                  Append_Line ("    val " & Tgt & " = " & Val & ".intersect(" & Args & ")");
               when Step_Struct_New =>
                  Append_Line ("    val " & Tgt & " = " & Args & "(" & Val & ")");
               when Step_Struct_Get =>
                  Append_Line ("    val " & Tgt & " = " & Val & "." & Args);
               when Step_Struct_Set =>
                  Append_Line ("    " & Tgt & "." & Args & " = " & Val);
               when Step_Generic_Call =>
                  Append_Line ("    val " & Tgt & " = " & Val & "<" & Args & ">()");
               when Step_Type_Cast =>
                  Append_Line ("    val " & Tgt & " = " & Val & " as " & Args);
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("    // unsupported step");
            end case;
         end;
         <<Continue_Kotlin>>
      end loop;
   end Emit_Steps_Kotlin;

   procedure Emit_Steps_SPARK (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      if Func.Steps.Count = 0 then
         Append_Line ("   return Integer'First;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_SPARK;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & ";");
                  else
                     Append_Line ("   --  assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
                  else
                     Append_Line ("   " & Val & " (" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("   return " & Val & ";");
                  else
                     Append_Line ("   return;");
                  end if;
               when Step_If =>
                  Append_Line ("   if " & Cond & " then");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("      return " & B_Val & ";");
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("   else");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                    else
                                       Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("      return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("      --  step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("   end if;");
               when Step_While =>
                  Append_Line ("   while " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_For =>
                  Append_Line ("   for I in " & Init & " .. " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_Break =>
                  Append_Line ("   exit;");
               when Step_Continue =>
                  Append_Line ("   goto Continue;");
               when Step_Error =>
                  Append_Line ("   raise Program_Error with """ & Val & """;");
               when Step_Switch =>
                  Append_Line ("   case " & Cond & " is");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("      when " & Case_Val & " =>");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("         " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("         return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("         --  step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("      when others => null;");
                  Append_Line ("   end case;");
               when Step_Try =>
                  Append_Line ("   begin");
                  Append_Line ("      " & Tgt & " := " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Var : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("   exception");
                           Append_Line ("      when others =>");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("         " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("         --  step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end;");
               when Step_Throw =>
                  Append_Line ("   raise Program_Error with """ & Val & """;");
               when Step_Array_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Vectors.Empty_Vector;");
               when Step_Array_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Element (" & Args & ");");
               when Step_Array_Set =>
                  Append_Line ("   " & Val & ".Replace_Element (" & Args & ", " & Tgt & ");");
               when Step_Array_Push =>
                  Append_Line ("   " & Tgt & ".Append (" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Last_Element;");
                  Append_Line ("   " & Val & ".Delete_Last;");
               when Step_Array_Len =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Length;");
               when Step_Map_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Maps.Empty_Map;");
               when Step_Map_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Element (" & Args & ");");
               when Step_Map_Set =>
                  Append_Line ("   " & Tgt & ".Insert (" & Args & ", " & Val & ");");
               when Step_Map_Delete =>
                  Append_Line ("   " & Tgt & ".Delete (" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Contains (" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("   --  map keys iteration for " & Val);
               when Step_Set_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Sets.Empty_Set;");
               when Step_Set_Add =>
                  Append_Line ("   " & Tgt & ".Insert (" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("   " & Tgt & ".Delete (" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Contains (" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("   --  set union: " & Tgt & " := " & Val & " union " & Args);
               when Step_Set_Intersect =>
                  Append_Line ("   --  set intersect: " & Tgt & " := " & Val & " intersect " & Args);
               when Step_Struct_New =>
                  Append_Line ("   " & Tgt & " := (" & Val & ");");
               when Step_Struct_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("   " & Tgt & "." & Args & " := " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("   " & Tgt & " := " & Args & " (" & Val & ");");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("   --  unsupported step");
            end case;
         end;
         <<Continue_SPARK>>
      end loop;
   end Emit_Steps_SPARK;

   procedure Emit_Steps_Ada (Func : IR_Function; Append_Line : not null access procedure (Text : String)) is
   begin
      --  Ada emitter is identical to SPARK for now
      --  Future: SPARK may add pragma SPARK_Mode, contracts, etc.
      if Func.Steps.Count = 0 then
         Append_Line ("   return Integer'First;");
         return;
      end if;

      for J in Step_Index range 1 .. Func.Steps.Count loop
         if Is_In_Nested_Block (Func, J) then
            goto Continue_Ada;
         end if;
         
         declare
            Step : constant IR_Step := Func.Steps.Steps (J);
            Val  : constant String := IStr.To_String (Step.Value);
            Tgt  : constant String := IStr.To_String (Step.Target);
            Cond : constant String := IStr.To_String (Step.Condition);
            Args : constant String := IStr.To_String (Step.Args);
            Init : constant String := IStr.To_String (Step.Init);
            Incr : constant String := IStr.To_String (Step.Increment);
         begin
            case Step.Step_Type is
               when Step_Assign =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & ";");
                  else
                     Append_Line ("   --  assign: " & Val);
                  end if;
               when Step_Call =>
                  if Tgt'Length > 0 then
                     Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
                  else
                     Append_Line ("   " & Val & " (" & Args & ");");
                  end if;
               when Step_Return =>
                  if Val'Length > 0 then
                     Append_Line ("   return " & Val & ";");
                  else
                     Append_Line ("   return;");
                  end if;
               when Step_If =>
                  Append_Line ("   if " & Cond & " then");
                  for B in Step_Index range Step.Then_Start .. Step.Then_Start + Step.Then_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when Step_Return =>
                                 Append_Line ("      return " & B_Val & ";");
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  if Step.Else_Count > 0 then
                     Append_Line ("   else");
                     for B in Step_Index range Step.Else_Start .. Step.Else_Start + Step.Else_Count - 1 loop
                        if B <= Func.Steps.Count then
                           declare
                              Body_Step : constant IR_Step := Func.Steps.Steps (B);
                              B_Val : constant String := IStr.To_String (Body_Step.Value);
                              B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                              B_Args : constant String := IStr.To_String (Body_Step.Args);
                           begin
                              case Body_Step.Step_Type is
                                 when Step_Assign =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                    end if;
                                 when Step_Call =>
                                    if B_Tgt'Length > 0 then
                                       Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                    else
                                       Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                    end if;
                                 when Step_Return =>
                                    Append_Line ("      return " & B_Val & ";");
                                 when others =>
                                    Append_Line ("      --  step");
                              end case;
                           end;
                        end if;
                     end loop;
                  end if;
                  Append_Line ("   end if;");
               when Step_While =>
                  Append_Line ("   while " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_For =>
                  Append_Line ("   for I in " & Init & " .. " & Cond & " loop");
                  for B in Step_Index range Step.Body_Start .. Step.Body_Start + Step.Body_Count - 1 loop
                     if B <= Func.Steps.Count then
                        declare
                           Body_Step : constant IR_Step := Func.Steps.Steps (B);
                           B_Val : constant String := IStr.To_String (Body_Step.Value);
                           B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                           B_Args : constant String := IStr.To_String (Body_Step.Args);
                        begin
                           case Body_Step.Step_Type is
                              when Step_Assign =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & ";");
                                 end if;
                              when Step_Call =>
                                 if B_Tgt'Length > 0 then
                                    Append_Line ("      " & B_Tgt & " := " & B_Val & " (" & B_Args & ");");
                                 else
                                    Append_Line ("      " & B_Val & " (" & B_Args & ");");
                                 end if;
                              when others =>
                                 Append_Line ("      --  step");
                           end case;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end loop;");
               when Step_Break =>
                  Append_Line ("   exit;");
               when Step_Continue =>
                  Append_Line ("   goto Continue;");
               when Step_Error =>
                  Append_Line ("   raise Program_Error with """ & Val & """;");
               when Step_Switch =>
                  Append_Line ("   case " & Cond & " is");
                  for C in Integer range 1 .. Integer (Step.Case_Count) loop
                     if C <= Max_Cases then
                        declare
                           Case_Val : constant String := IStr.To_String (Step.Cases (C).Case_Value);
                        begin
                           Append_Line ("      when " & Case_Val & " =>");
                           for B in Step_Index range Step.Cases (C).Body_Start .. Step.Cases (C).Body_Start + Step.Cases (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("         " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when Step_Return =>
                                          Append_Line ("         return " & B_Val & ";");
                                       when others =>
                                          Append_Line ("         --  step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("      when others => null;");
                  Append_Line ("   end case;");
               when Step_Try =>
                  Append_Line ("   begin");
                  Append_Line ("      " & Tgt & " := " & Val & ";");
                  for C in Integer range 1 .. Integer (Step.Catch_Count) loop
                     if C <= Max_Catch_Blocks then
                        declare
                           Catch_Var : constant String := IStr.To_String (Step.Catch_Blocks (C).Var_Name);
                        begin
                           Append_Line ("   exception");
                           Append_Line ("      when others =>");
                           for B in Step_Index range Step.Catch_Blocks (C).Body_Start .. Step.Catch_Blocks (C).Body_Start + Step.Catch_Blocks (C).Body_Count - 1 loop
                              if B <= Func.Steps.Count then
                                 declare
                                    Body_Step : constant IR_Step := Func.Steps.Steps (B);
                                    B_Val : constant String := IStr.To_String (Body_Step.Value);
                                    B_Tgt : constant String := IStr.To_String (Body_Step.Target);
                                 begin
                                    case Body_Step.Step_Type is
                                       when Step_Assign =>
                                          if B_Tgt'Length > 0 then
                                             Append_Line ("         " & B_Tgt & " := " & B_Val & ";");
                                          end if;
                                       when others =>
                                          Append_Line ("         --  step;");
                                    end case;
                                 end;
                              end if;
                           end loop;
                        end;
                     end if;
                  end loop;
                  Append_Line ("   end;");
               when Step_Throw =>
                  Append_Line ("   raise Program_Error with """ & Val & """;");
               when Step_Array_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Vectors.Empty_Vector;");
               when Step_Array_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Element (" & Args & ");");
               when Step_Array_Set =>
                  Append_Line ("   " & Val & ".Replace_Element (" & Args & ", " & Tgt & ");");
               when Step_Array_Push =>
                  Append_Line ("   " & Tgt & ".Append (" & Val & ");");
               when Step_Array_Pop =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Last_Element;");
                  Append_Line ("   " & Val & ".Delete_Last;");
               when Step_Array_Len =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Length;");
               when Step_Map_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Maps.Empty_Map;");
               when Step_Map_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Element (" & Args & ");");
               when Step_Map_Set =>
                  Append_Line ("   " & Tgt & ".Insert (" & Args & ", " & Val & ");");
               when Step_Map_Delete =>
                  Append_Line ("   " & Tgt & ".Delete (" & Val & ");");
               when Step_Map_Has =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Contains (" & Args & ");");
               when Step_Map_Keys =>
                  Append_Line ("   --  map keys iteration for " & Val);
               when Step_Set_New =>
                  Append_Line ("   " & Tgt & " := " & Args & "_Sets.Empty_Set;");
               when Step_Set_Add =>
                  Append_Line ("   " & Tgt & ".Insert (" & Val & ");");
               when Step_Set_Remove =>
                  Append_Line ("   " & Tgt & ".Delete (" & Val & ");");
               when Step_Set_Has =>
                  Append_Line ("   " & Tgt & " := " & Val & ".Contains (" & Args & ");");
               when Step_Set_Union =>
                  Append_Line ("   --  set union: " & Tgt & " := " & Val & " union " & Args);
               when Step_Set_Intersect =>
                  Append_Line ("   --  set intersect: " & Tgt & " := " & Val & " intersect " & Args);
               when Step_Struct_New =>
                  Append_Line ("   " & Tgt & " := (" & Val & ");");
               when Step_Struct_Get =>
                  Append_Line ("   " & Tgt & " := " & Val & "." & Args & ";");
               when Step_Struct_Set =>
                  Append_Line ("   " & Tgt & "." & Args & " := " & Val & ";");
               when Step_Generic_Call =>
                  Append_Line ("   " & Tgt & " := " & Val & " (" & Args & ");");
               when Step_Type_Cast =>
                  Append_Line ("   " & Tgt & " := " & Args & " (" & Val & ");");
               when Step_Nop =>
                  null;  --  No operation
               when others =>
                  Append_Line ("   --  unsupported step");
            end case;
         end;
         <<Continue_Ada>>
      end loop;
   end Emit_Steps_Ada;

end Emit_Target.Mainstream;
