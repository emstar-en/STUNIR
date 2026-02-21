--  STUNIR DO-331 Model IR Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Model_IR is

   --  ============================================================
   --  DAL Requirements
   --  ============================================================

   function Get_DAL_Requirements (Level : DAL_Level) return DAL_Coverage_Requirement is
   begin
      case Level is
         when DAL_A =>
            return (Requires_MCDC               => True,
                    Requires_Decision_Coverage  => True,
                    Requires_Statement_Coverage => True,
                    Requires_State_Coverage     => True,
                    Requires_Transition_Coverage => True);

         when DAL_B =>
            return (Requires_MCDC               => False,
                    Requires_Decision_Coverage  => True,
                    Requires_Statement_Coverage => True,
                    Requires_State_Coverage     => True,
                    Requires_Transition_Coverage => True);

         when DAL_C =>
            return (Requires_MCDC               => False,
                    Requires_Decision_Coverage  => False,
                    Requires_Statement_Coverage => True,
                    Requires_State_Coverage     => True,
                    Requires_Transition_Coverage => True);

         when DAL_D =>
            return (Requires_MCDC               => False,
                    Requires_Decision_Coverage  => False,
                    Requires_Statement_Coverage => False,
                    Requires_State_Coverage     => False,
                    Requires_Transition_Coverage => False);

         when DAL_E =>
            return (Requires_MCDC               => False,
                    Requires_Decision_Coverage  => False,
                    Requires_Statement_Coverage => False,
                    Requires_State_Coverage     => False,
                    Requires_Transition_Coverage => False);
      end case;
   end Get_DAL_Requirements;

   --  ============================================================
   --  Element Operations
   --  ============================================================

   function Create_Element (
      Kind   : Element_Kind;
      Name   : String;
      Parent : Element_ID := Null_Element_ID
   ) return Model_Element
   is
      Result : Model_Element;
   begin
      Result.ID := Generate_ID;
      Result.Kind := Kind;
      Result.Name := (others => ' ');
      Result.Name (1 .. Name'Length) := Name;
      Result.Name_Length := Name'Length;
      Result.Parent_ID := Parent;
      Result.Child_Count := 0;
      Result.Line_Number := 0;
      Result.Is_Abstract := False;
      Result.Is_Root := Parent = Null_Element_ID;
      return Result;
   end Create_Element;

   function Get_Name (Element : Model_Element) return String is
   begin
      return Element.Name (1 .. Element.Name_Length);
   end Get_Name;

   procedure Set_Name (
      Element : in Out Model_Element;
      Name    : in     String
   ) is
   begin
      Element.Name := (others => ' ');
      Element.Name (1 .. Name'Length) := Name;
      Element.Name_Length := Name'Length;
   end Set_Name;

   function Is_Valid (Element : Model_Element) return Boolean is
   begin
      return Element.ID /= Null_Element_ID
             and Element.Name_Length > 0
             and Element.Name_Length <= Max_Name_Length;
   end Is_Valid;

   function Is_Structural (Kind : Element_Kind) return Boolean is
   begin
      return Kind in Package_Element | Part_Element | Attribute_Element |
                     Connection_Element | Port_Element | Interface_Element;
   end Is_Structural;

   function Is_Behavioral (Kind : Element_Kind) return Boolean is
   begin
      return Kind in Action_Element | State_Element | Transition_Element |
                     Activity_Element;
   end Is_Behavioral;

   --  ============================================================
   --  Container Operations
   --  ============================================================

   function Create_Container return Model_Container is
   begin
      return Null_Container;
   end Create_Container;

   procedure Set_IR_Hash (
      Container : in out Model_Container;
      Hash      : in     String
   ) is
   begin
      Container.IR_Source_Hash := (others => '0');
      Container.IR_Source_Hash (1 .. Hash'Length) := Hash;
      Container.Hash_Length := Hash'Length;
   end Set_IR_Hash;

   procedure Set_Module_Name (
      Container : in Out Model_Container;
      Name      : in     String
   ) is
   begin
      Container.Module_Name := (others => ' ');
      Container.Module_Name (1 .. Name'Length) := Name;
      Container.Module_Name_Len := Name'Length;
   end Set_Module_Name;

   --  ============================================================
   --  ID Generation
   --  ============================================================

   function Generate_ID return Element_ID is
      Result : Element_ID;
   begin
      Result := Next_ID;
      if Next_ID < Element_ID'Last then
         Next_ID := Next_ID + 1;
      end if;
      return Result;
   end Generate_ID;

   procedure Reset_ID_Generator is
   begin
      Next_ID := 1;
   end Reset_ID_Generator;

end Model_IR;
