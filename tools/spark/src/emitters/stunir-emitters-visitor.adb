-- STUNIR Visitor Pattern Implementation
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.Visitor is
   pragma SPARK_Mode (On);

   function Null_Node_ID return IR.Types.ID is
     (IR.Types.Name_Strings.Null_Bounded_String);

   procedure Traverse_Module
     (Module  : in     IR.Modules.IR_Module;
      Nodes   : in     STUNIR.Emitters.Node_Table.Node_Table;
      Context : in out Visitor_Context'Class;
      Result  :    out Visit_Result)
   is
   begin
      Result := Continue;
      Context.Result := Continue;

      -- Start module visit
      On_Module_Start (Context, Module);
      if Context.Result = Abort_Visit then
         Result := Abort_Visit;
         return;
      end if;

      -- Traverse declarations via node table resolution.
      for I in 1 .. Module.Decl_Count loop
         declare
            Decl_ID : constant IR.Types.ID := Module.Declarations (I);
            Decl_Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
              STUNIR.Emitters.Node_Table.Lookup (Nodes, Decl_ID);
         begin
            if Decl_Index = 0 then
               null;
            else
               declare
                  Decl_Node : constant STUNIR.Emitters.Node_Table.Declaration_Record :=
                    STUNIR.Emitters.Node_Table.Get_Declaration (Nodes, Decl_Index);
               begin
                  case Decl_Node.Kind is
                     when IR.Types.Kind_Type_Decl =>
                        declare
                           T : IR.Declarations.Type_Declaration := Decl_Node.Typ;
                        begin
                           On_Type_Start (Context, T);
                           if Context.Result = Abort_Visit then
                              Result := Abort_Visit;
                              return;
                           end if;
                           On_Type_End (Context, T);
                           if Context.Result = Abort_Visit then
                              Result := Abort_Visit;
                              return;
                           end if;
                        end;

                     when IR.Types.Kind_Function_Decl =>
                        declare
                           F : IR.Declarations.Function_Declaration := Decl_Node.Func;
                        begin
                           On_Function_Start (Context, F);
                           if Context.Result = Abort_Visit then
                              Result := Abort_Visit;
                              return;
                           end if;

                           if IR.Nodes.Is_Valid_Node_ID (F.Body_ID) then
                              declare
                                 Stmt_Index : constant STUNIR.Emitters.Node_Table.Node_Index :=
                                   STUNIR.Emitters.Node_Table.Lookup (Nodes, F.Body_ID);
                              begin
                                 if Stmt_Index > 0 then
                                    declare
                                       Stmt_Node : constant STUNIR.Emitters.Node_Table.Statement_Record :=
                                         STUNIR.Emitters.Node_Table.Get_Statement (Nodes, Stmt_Index);
                                    begin
                                       On_Statement (Context, Stmt_Node.Base);
                                       if Context.Result = Abort_Visit then
                                          Result := Abort_Visit;
                                          return;
                                       end if;
                                    end;
                                 end if;
                              end;
                           end if;

                           On_Function_End (Context, F);
                           if Context.Result = Abort_Visit then
                              Result := Abort_Visit;
                              return;
                           end if;
                        end;

                     when others =>
                        null;
                  end case;
               end;
            end if;
         end;
      end loop;

      -- End module visit
      On_Module_End (Context, Module);
      if Context.Result = Abort_Visit then
         Result := Abort_Visit;
         return;
      end if;

      Result := Continue;
   end Traverse_Module;

end STUNIR.Emitters.Visitor;
