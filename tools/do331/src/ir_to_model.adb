--  STUNIR DO-331 IR-to-Model Transformer Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Transformer_Utils; use Transformer_Utils;

package body IR_To_Model is

   --  ============================================================
   --  Storage Operations
   --  ============================================================

   procedure Initialize_Storage (Storage : out Element_Storage) is
   begin
      Storage.Count := 0;
      Storage.Action_Count := 0;
      Storage.State_Count := 0;
      Storage.Trans_Count := 0;
      
      --  Initialize arrays to null values
      for I in Storage.Elements'Range loop
         Storage.Elements (I) := Null_Element;
      end loop;
      
      for I in Storage.Action_Data'Range loop
         Storage.Action_Data (I) := Null_Action_Data;
      end loop;
      
      for I in Storage.State_Data'Range loop
         Storage.State_Data (I) := Null_State_Data;
      end loop;
      
      for I in Storage.Trans_Data'Range loop
         Storage.Trans_Data (I) := Null_Transition_Data;
      end loop;
   end Initialize_Storage;

   --  ============================================================
   --  Transformation Procedures
   --  ============================================================

   procedure Transform_Module (
      Module_Name : in     String;
      IR_Hash     : in     String;
      Options     : in     Transform_Options;
      Container   : in Out Model_Container;
      Storage     : in Out Element_Storage;
      Root_ID     : out    Element_ID
   ) is
      Elem : Model_Element;
   begin
      --  Set container metadata
      Container.Schema_Version := 1;
      Container.DAL_Level := Options.DAL_Level;
      Container.Generation_Epoch := Get_Current_Epoch;
      
      if IR_Hash'Length > 0 and IR_Hash'Length <= Max_Hash_Length then
         Set_IR_Hash (Container, IR_Hash);
      end if;
      
      Set_Module_Name (Container, Module_Name);
      
      --  Create root package element
      Elem := Create_Element (
         Kind   => Package_Element,
         Name   => To_SysML_Identifier (Module_Name),
         Parent => Null_Element_ID
      );
      Elem.Is_Root := True;
      
      --  Add to storage
      if Storage.Count < Max_Elements then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         Container.Element_Count := Storage.Count;
         Root_ID := Elem.ID;
      else
         Root_ID := Null_Element_ID;
      end if;
   end Transform_Module;

   procedure Transform_Function (
      Func_Name   : in     String;
      Parent_ID   : in     Element_ID;
      Has_Inputs  : in     Boolean;
      Has_Outputs : in     Boolean;
      Input_Count : in     Natural;
      Output_Count: in     Natural;
      Storage     : in Out Element_Storage;
      Action_ID   : out    Element_ID
   ) is
      Elem : Model_Element;
      Data : Action_Data;
   begin
      --  Create action element
      Elem := Create_Element (
         Kind   => Action_Element,
         Name   => To_SysML_Identifier (Func_Name),
         Parent => Parent_ID
      );
      
      --  Create action data
      Data := (
         Has_Inputs   => Has_Inputs,
         Has_Outputs  => Has_Outputs,
         Input_Count  => Input_Count,
         Output_Count => Output_Count,
         Is_Entry     => False,
         Is_Exit      => False
      );
      
      --  Add to storage
      if Storage.Count < Max_Elements and Storage.Action_Count < Max_Actions then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         
         Storage.Action_Count := Storage.Action_Count + 1;
         Storage.Action_Data (Storage.Action_Count) := Data;
         
         Action_ID := Elem.ID;
         
         --  Update parent's child count
         for I in 1 .. Storage.Count - 1 loop
            if Storage.Elements (I).ID = Parent_ID then
               Storage.Elements (I).Child_Count := 
                  Storage.Elements (I).Child_Count + 1;
               exit;
            end if;
         end loop;
      else
         Action_ID := Null_Element_ID;
      end if;
   end Transform_Function;

   procedure Transform_Type (
      Type_Name   : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Attr_ID     : out    Element_ID
   ) is
      Elem : Model_Element;
   begin
      Elem := Create_Element (
         Kind   => Attribute_Element,
         Name   => To_SysML_Identifier (Type_Name),
         Parent => Parent_ID
      );
      
      if Storage.Count < Max_Elements then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         Attr_ID := Elem.ID;
         
         --  Update parent's child count
         for I in 1 .. Storage.Count - 1 loop
            if Storage.Elements (I).ID = Parent_ID then
               Storage.Elements (I).Child_Count := 
                  Storage.Elements (I).Child_Count + 1;
               exit;
            end if;
         end loop;
      else
         Attr_ID := Null_Element_ID;
      end if;
   end Transform_Type;

   procedure Transform_Variable (
      Var_Name    : in     String;
      Var_Type    : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Var_ID      : out    Element_ID
   ) is
      Elem : Model_Element;
      pragma Unreferenced (Var_Type);  -- Used for documentation
   begin
      Elem := Create_Element (
         Kind   => Attribute_Element,
         Name   => To_SysML_Identifier (Var_Name),
         Parent => Parent_ID
      );
      
      if Storage.Count < Max_Elements then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         Var_ID := Elem.ID;
      else
         Var_ID := Null_Element_ID;
      end if;
   end Transform_Variable;

   procedure Add_State (
      State_Name  : in     String;
      Parent_ID   : in     Element_ID;
      Is_Initial  : in     Boolean;
      Is_Final    : in     Boolean;
      Storage     : in Out Element_Storage;
      State_ID    : out    Element_ID
   ) is
      Elem : Model_Element;
      Data : State_Data;
   begin
      Elem := Create_Element (
         Kind   => State_Element,
         Name   => To_SysML_Identifier (State_Name),
         Parent => Parent_ID
      );
      
      Data := Null_State_Data;
      Data.Is_Initial := Is_Initial;
      Data.Is_Final := Is_Final;
      
      if Is_Initial then
         Data.State_Type := Initial_State;
      elsif Is_Final then
         Data.State_Type := Final_State;
      else
         Data.State_Type := Simple_State;
      end if;
      
      if Storage.Count < Max_Elements and Storage.State_Count < Max_States then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         
         Storage.State_Count := Storage.State_Count + 1;
         Storage.State_Data (Storage.State_Count) := Data;
         
         State_ID := Elem.ID;
      else
         State_ID := Null_Element_ID;
      end if;
   end Add_State;

   procedure Add_Transition (
      Trans_Name  : in     String;
      Source_ID   : in     Element_ID;
      Target_ID   : in     Element_ID;
      Has_Guard   : in     Boolean;
      Guard_Expr  : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Trans_ID    : out    Element_ID
   ) is
      Elem : Model_Element;
      Data : Transition_Data;
      Name_To_Use : String (1 .. Max_Name_Length) := (others => ' ');
      Name_Len : Natural;
   begin
      --  Generate name if not provided
      if Trans_Name'Length = 0 then
         Name_To_Use (1 .. 10) := "transition";
         Name_Len := 10;
      else
         Name_Len := Natural'Min (Trans_Name'Length, Max_Name_Length);
         Name_To_Use (1 .. Name_Len) := Trans_Name (Trans_Name'First .. Trans_Name'First + Name_Len - 1);
      end if;
      
      Elem := Create_Element (
         Kind   => Transition_Element,
         Name   => Name_To_Use (1 .. Name_Len),
         Parent => Parent_ID
      );
      
      Data := Null_Transition_Data;
      Data.Source_State_ID := Source_ID;
      Data.Target_State_ID := Target_ID;
      Data.Has_Guard := Has_Guard;
      
      if Has_Guard and Guard_Expr'Length > 0 then
         declare
            Len : constant Natural := Natural'Min (Guard_Expr'Length, Max_Name_Length);
         begin
            Data.Guard_Expr (1 .. Len) := Guard_Expr (Guard_Expr'First .. Guard_Expr'First + Len - 1);
            Data.Guard_Length := Len;
         end;
      end if;
      
      if Storage.Count < Max_Elements and Storage.Trans_Count < Max_Transitions then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         
         Storage.Trans_Count := Storage.Trans_Count + 1;
         Storage.Trans_Data (Storage.Trans_Count) := Data;
         
         Trans_ID := Elem.ID;
      else
         Trans_ID := Null_Element_ID;
      end if;
   end Add_Transition;

   procedure Add_Requirement (
      Req_Name    : in     String;
      Req_Text    : in     String;
      Parent_ID   : in     Element_ID;
      Storage     : in Out Element_Storage;
      Req_ID      : out    Element_ID
   ) is
      Elem : Model_Element;
      pragma Unreferenced (Req_Text);  -- Stored in documentation
   begin
      Elem := Create_Element (
         Kind   => Requirement_Element,
         Name   => To_SysML_Identifier (Req_Name),
         Parent => Parent_ID
      );
      
      if Storage.Count < Max_Elements then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         Req_ID := Elem.ID;
      else
         Req_ID := Null_Element_ID;
      end if;
   end Add_Requirement;

   procedure Add_Satisfy (
      Req_ID      : in     Element_ID;
      Element_ID  : in     Model_IR.Element_ID;
      Parent_ID   : in     Model_IR.Element_ID;
      Storage     : in Out Element_Storage;
      Sat_ID      : out    Model_IR.Element_ID
   ) is
      Elem : Model_Element;
      pragma Unreferenced (Req_ID, Element_ID);  -- For traceability
   begin
      Elem := Create_Element (
         Kind   => Satisfy_Element,
         Name   => "satisfy",
         Parent => Parent_ID
      );
      
      if Storage.Count < Max_Elements then
         Storage.Count := Storage.Count + 1;
         Storage.Elements (Storage.Count) := Elem;
         Sat_ID := Elem.ID;
      else
         Sat_ID := Null_Element_ID;
      end if;
   end Add_Satisfy;

   --  ============================================================
   --  Lookup Operations
   --  ============================================================

   function Get_Element (
      Storage : Element_Storage;
      ID      : Element_ID
   ) return Model_Element is
   begin
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).ID = ID then
            return Storage.Elements (I);
         end if;
      end loop;
      return Null_Element;
   end Get_Element;

   function Get_Action_Data (
      Storage   : Element_Storage;
      Action_ID : Element_ID
   ) return Action_Data is
      Action_Idx : Natural := 0;
   begin
      --  Find the action index
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).Kind = Action_Element then
            Action_Idx := Action_Idx + 1;
            if Storage.Elements (I).ID = Action_ID then
               if Action_Idx <= Storage.Action_Count then
                  return Storage.Action_Data (Action_Idx);
               else
                  return Null_Action_Data;
               end if;
            end if;
         end if;
      end loop;
      return Null_Action_Data;
   end Get_Action_Data;

   function Get_State_Data (
      Storage  : Element_Storage;
      State_ID : Element_ID
   ) return State_Data is
      State_Idx : Natural := 0;
   begin
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).Kind = State_Element then
            State_Idx := State_Idx + 1;
            if Storage.Elements (I).ID = State_ID then
               if State_Idx <= Storage.State_Count then
                  return Storage.State_Data (State_Idx);
               else
                  return Null_State_Data;
               end if;
            end if;
         end if;
      end loop;
      return Null_State_Data;
   end Get_State_Data;

   function Get_Transition_Data (
      Storage  : Element_Storage;
      Trans_ID : Element_ID
   ) return Transition_Data is
      Trans_Idx : Natural := 0;
   begin
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).Kind = Transition_Element then
            Trans_Idx := Trans_Idx + 1;
            if Storage.Elements (I).ID = Trans_ID then
               if Trans_Idx <= Storage.Trans_Count then
                  return Storage.Trans_Data (Trans_Idx);
               else
                  return Null_Transition_Data;
               end if;
            end if;
         end if;
      end loop;
      return Null_Transition_Data;
   end Get_Transition_Data;

   function Get_Children (
      Storage   : Element_Storage;
      Parent_ID : Element_ID
   ) return Element_ID_Array is
      Count : Natural := 0;
   begin
      --  Count children
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).Parent_ID = Parent_ID then
            Count := Count + 1;
         end if;
      end loop;
      
      --  Build result array
      declare
         Result : Element_ID_Array (1 .. Count);
         J      : Natural := 0;
      begin
         for I in 1 .. Storage.Count loop
            if Storage.Elements (I).Parent_ID = Parent_ID then
               J := J + 1;
               Result (J) := Storage.Elements (I).ID;
            end if;
         end loop;
         return Result;
      end;
   end Get_Children;

   --  ============================================================
   --  Rule Information
   --  ============================================================

   function Get_Rule_Name (Rule : Transformation_Rule) return String is
   begin
      case Rule is
         when Rule_Module_To_Package     => return "Module to Package";
         when Rule_Function_To_Action    => return "Function to Action Definition";
         when Rule_Type_To_Attribute_Def => return "Type to Attribute Definition";
         when Rule_Variable_To_Attribute => return "Variable to Attribute";
         when Rule_If_To_Decision        => return "If Statement to Decision";
         when Rule_Loop_To_Action        => return "Loop to Loop Action";
         when Rule_Return_To_Output      => return "Return to Output Assignment";
         when Rule_Call_To_Perform       => return "Function Call to Perform Action";
         when Rule_Import_To_Import      => return "Import to Import Statement";
         when Rule_Export_To_Expose      => return "Export to Expose Statement";
         when Rule_State_Machine         => return "State Machine Definition";
         when Rule_Transition            => return "State Transition";
         when Rule_Requirement           => return "Requirement Definition";
         when Rule_Constraint            => return "Constraint Definition";
         when Rule_Satisfy_Relationship  => return "Satisfy Relationship";
      end case;
   end Get_Rule_Name;

   function Get_DO331_Objective (Rule : Transformation_Rule) return String is
   begin
      case Rule is
         when Rule_Module_To_Package     => return "MB.2";
         when Rule_Function_To_Action    => return "MB.3";
         when Rule_Type_To_Attribute_Def => return "MB.2";
         when Rule_Variable_To_Attribute => return "MB.3";
         when Rule_If_To_Decision        => return "MB.3";
         when Rule_Loop_To_Action        => return "MB.3";
         when Rule_Return_To_Output      => return "MB.3";
         when Rule_Call_To_Perform       => return "MB.3";
         when Rule_Import_To_Import      => return "MB.2";
         when Rule_Export_To_Expose      => return "MB.2";
         when Rule_State_Machine         => return "MB.3";
         when Rule_Transition            => return "MB.3";
         when Rule_Requirement           => return "MB.1";
         when Rule_Constraint            => return "MB.4";
         when Rule_Satisfy_Relationship  => return "MB.6";
      end case;
   end Get_DO331_Objective;

end IR_To_Model;
