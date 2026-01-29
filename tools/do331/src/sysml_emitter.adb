--  STUNIR DO-331 SysML 2.0 Emitter Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with SysML_Types; use SysML_Types;
with Transformer_Utils; use Transformer_Utils;

package body SysML_Emitter is

   --  ============================================================
   --  Buffer Operations
   --  ============================================================

   procedure Initialize_Buffer (Buffer : out Output_Buffer) is
   begin
      Buffer.Data := (others => ' ');
      Buffer.Length := 0;
      Buffer.Line_Count := 0;
   end Initialize_Buffer;

   procedure Append (
      Buffer : in Out Output_Buffer;
      Text   : in     String
   ) is
   begin
      if Buffer.Length + Text'Length <= Max_Output_Length then
         Buffer.Data (Buffer.Length + 1 .. Buffer.Length + Text'Length) := Text;
         Buffer.Length := Buffer.Length + Text'Length;
      end if;
   end Append;

   procedure Append_Line (
      Buffer : in Out Output_Buffer;
      Text   : in     String
   ) is
   begin
      Append (Buffer, Text);
      New_Line (Buffer);
   end Append_Line;

   procedure New_Line (Buffer : in out Output_Buffer) is
   begin
      if Buffer.Length < Max_Output_Length then
         Buffer.Length := Buffer.Length + 1;
         Buffer.Data (Buffer.Length) := ASCII.LF;
         Buffer.Line_Count := Buffer.Line_Count + 1;
      end if;
   end New_Line;

   function Get_Content (Buffer : Output_Buffer) return String is
   begin
      if Buffer.Length > 0 then
         return Buffer.Data (1 .. Buffer.Length);
      else
         return "";
      end if;
   end Get_Content;

   --  ============================================================
   --  Helper Functions
   --  ============================================================

   function Indent_String (Level : Natural; Size : Positive) return String is
      Result : String (1 .. Level * Size) := (others => ' ');
   begin
      return Result;
   end Indent_String;

   --  ============================================================
   --  Main Emission
   --  ============================================================

   procedure Emit_Model (
      Container : in     Model_Container;
      Storage   : in     Element_Storage;
      Options   : in     Emitter_Options;
      Buffer    : in Out Output_Buffer;
      Result    : out    Emitter_Result
   ) is
      Element_Emitted : Natural := 0;
   begin
      Initialize_Buffer (Buffer);
      
      --  Emit header with metadata
      if Options.Include_Metadata then
         Emit_Header (Container, Options, Buffer);
      end if;
      
      --  Emit all root elements (packages)
      for I in 1 .. Storage.Count loop
         if Storage.Elements (I).Is_Root then
            Emit_Package (
               Element => Storage.Elements (I),
               Storage => Storage,
               Options => Options,
               Indent  => 0,
               Buffer  => Buffer
            );
            Element_Emitted := Element_Emitted + 1;
         end if;
      end loop;
      
      --  Set result
      if Buffer.Length > 0 then
         Result := (
            Status        => Emit_Success,
            Output_Length => Buffer.Length,
            Line_Count    => Buffer.Line_Count,
            Element_Count => Element_Emitted
         );
      else
         Result := (
            Status        => Emit_Error,
            Output_Length => 0,
            Line_Count    => 0,
            Element_Count => 0
         );
      end if;
   end Emit_Model;

   --  ============================================================
   --  Header Emission
   --  ============================================================

   procedure Emit_Header (
      Container : in     Model_Container;
      Options   : in     Emitter_Options;
      Buffer    : in Out Output_Buffer
   ) is
      pragma Unreferenced (Options);
   begin
      --  Emit documentation comment with metadata
      Append_Line (Buffer, "/* STUNIR DO-331 Model-Based Development Output");
      Append_Line (Buffer, " * Schema Version: " & Natural_To_String (Container.Schema_Version));
      
      if Container.Hash_Length > 0 then
         Append (Buffer, " * Source IR Hash: ");
         Append_Line (Buffer, Container.IR_Source_Hash (1 .. Container.Hash_Length));
      end if;
      
      Append_Line (Buffer, " * Generation Epoch: " & Natural_To_String (Container.Generation_Epoch));
      
      Append (Buffer, " * DAL Level: ");
      case Container.DAL_Level is
         when DAL_A => Append_Line (Buffer, "A");
         when DAL_B => Append_Line (Buffer, "B");
         when DAL_C => Append_Line (Buffer, "C");
         when DAL_D => Append_Line (Buffer, "D");
         when DAL_E => Append_Line (Buffer, "E");
      end case;
      
      Append_Line (Buffer, " */");
      New_Line (Buffer);
      
      --  Standard imports
      Append_Line (Buffer, "import ScalarValues::*;");
      New_Line (Buffer);
   end Emit_Header;

   --  ============================================================
   --  Package Emission
   --  ============================================================

   procedure Emit_Package (
      Element : in     Model_Element;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
      Children : constant Element_ID_Array := Get_Children (Storage, Element.ID);
   begin
      --  Package declaration
      Append (Buffer, Ind);
      Append (Buffer, "package ");
      Append (Buffer, Get_Name (Element));
      Append_Line (Buffer, " {");
      
      --  Documentation comment
      if Options.Include_Comments then
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append (Buffer, "doc /* Module: ");
         Append (Buffer, Get_Name (Element));
         Append_Line (Buffer, " */");
         New_Line (Buffer);
      end if;
      
      --  Emit children
      for I in Children'Range loop
         declare
            Child : constant Model_Element := Get_Element (Storage, Children (I));
         begin
            case Child.Kind is
               when Package_Element =>
                  Emit_Package (Child, Storage, Options, Indent + 1, Buffer);
               
               when Action_Element =>
                  Emit_Action_Def (
                     Element => Child,
                     Data    => Get_Action_Data (Storage, Child.ID),
                     Storage => Storage,
                     Options => Options,
                     Indent  => Indent + 1,
                     Buffer  => Buffer
                  );
               
               when Attribute_Element =>
                  Emit_Attribute (Child, Options, Indent + 1, Buffer);
               
               when State_Element =>
                  Emit_State (
                     Element => Child,
                     Data    => Get_State_Data (Storage, Child.ID),
                     Options => Options,
                     Indent  => Indent + 1,
                     Buffer  => Buffer
                  );
               
               when Transition_Element =>
                  Emit_Transition (
                     Element => Child,
                     Data    => Get_Transition_Data (Storage, Child.ID),
                     Storage => Storage,
                     Options => Options,
                     Indent  => Indent + 1,
                     Buffer  => Buffer
                  );
               
               when Requirement_Element =>
                  Emit_Requirement (Child, Options, Indent + 1, Buffer);
               
               when Satisfy_Element =>
                  Emit_Satisfy (Child, Options, Indent + 1, Buffer);
               
               when others =>
                  null;  -- Skip other element types
            end case;
         end;
      end loop;
      
      --  Close package
      Append (Buffer, Ind);
      Append_Line (Buffer, "}");
      New_Line (Buffer);
   end Emit_Package;

   --  ============================================================
   --  Action Emission
   --  ============================================================

   procedure Emit_Action_Def (
      Element : in     Model_Element;
      Data    : in     Action_Data;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
      pragma Unreferenced (Storage);
   begin
      --  Action definition
      Append (Buffer, Ind);
      Append (Buffer, "action def ");
      Append (Buffer, Get_Name (Element));
      Append_Line (Buffer, " {");
      
      --  Input parameters
      if Data.Has_Inputs then
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append_Line (Buffer, "in inputs : Anything;  // Input parameters");
      end if;
      
      --  Output parameters
      if Data.Has_Outputs then
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append_Line (Buffer, "out result : Anything;  // Output");
      end if;
      
      --  Coverage point: Entry
      if Options.Include_Coverage then
         New_Line (Buffer);
         Emit_Coverage_Point ("CP_ENTRY", "entry", Indent + 1, Buffer);
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append_Line (Buffer, "first start;");
      end if;
      
      --  Coverage point: Exit
      if Options.Include_Coverage then
         New_Line (Buffer);
         Emit_Coverage_Point ("CP_EXIT", "exit", Indent + 1, Buffer);
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append_Line (Buffer, "then done;");
      end if;
      
      Append (Buffer, Ind);
      Append_Line (Buffer, "}");
      New_Line (Buffer);
   end Emit_Action_Def;

   --  ============================================================
   --  Attribute Emission
   --  ============================================================

   procedure Emit_Attribute (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
   begin
      Append (Buffer, Ind);
      Append (Buffer, "attribute def ");
      Append (Buffer, Get_Name (Element));
      Append_Line (Buffer, " :> Real {");
      
      if Options.Include_Comments then
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append (Buffer, "doc /* Type: ");
         Append (Buffer, Get_Name (Element));
         Append_Line (Buffer, " */");
      end if;
      
      Append (Buffer, Ind);
      Append_Line (Buffer, "}");
      New_Line (Buffer);
   end Emit_Attribute;

   --  ============================================================
   --  State Emission
   --  ============================================================

   procedure Emit_State (
      Element : in     Model_Element;
      Data    : in     State_Data;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
   begin
      Append (Buffer, Ind);
      
      --  Emit state keyword based on type
      if Data.Is_Initial then
         Append (Buffer, "entry state ");
      elsif Data.Is_Final then
         Append (Buffer, "state ");
      else
         Append (Buffer, "state ");
      end if;
      
      Append (Buffer, Get_Name (Element));
      
      --  Check for entry/exit actions
      if Data.Has_Entry_Action or Data.Has_Exit_Action or Options.Include_Coverage then
         Append_Line (Buffer, " {");
         
         --  Coverage point for state
         if Options.Include_Coverage then
            declare
               Point_ID : constant String := "CP_STATE_" & Get_Name (Element);
            begin
               Emit_Coverage_Point (Point_ID, "state_coverage", Indent + 1, Buffer);
            end;
         end if;
         
         if Data.Has_Entry_Action then
            Append (Buffer, Ind);
            Append (Buffer, "    ");
            Append_Line (Buffer, "entry action { /* entry action */ }");
         end if;
         
         if Data.Has_Exit_Action then
            Append (Buffer, Ind);
            Append (Buffer, "    ");
            Append_Line (Buffer, "exit action { /* exit action */ }");
         end if;
         
         Append (Buffer, Ind);
         Append_Line (Buffer, "}");
      else
         Append_Line (Buffer, ";");
      end if;
      
      New_Line (Buffer);
   end Emit_State;

   --  ============================================================
   --  Transition Emission
   --  ============================================================

   procedure Emit_Transition (
      Element : in     Model_Element;
      Data    : in     Transition_Data;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
      Source : Model_Element;
      Target : Model_Element;
   begin
      --  Coverage point for transition
      if Options.Include_Coverage then
         declare
            Point_ID : constant String := "CP_TRANS_" & Get_Name (Element);
         begin
            Emit_Coverage_Point (Point_ID, "transition_coverage", Indent, Buffer);
         end;
      end if;
      
      Append (Buffer, Ind);
      Append (Buffer, "transition ");
      
      --  Get source and target state names
      Source := Get_Element (Storage, Data.Source_State_ID);
      Target := Get_Element (Storage, Data.Target_State_ID);
      
      if Is_Valid (Source) and Is_Valid (Target) then
         Append (Buffer, Get_Name (Source));
         Append (Buffer, " -> ");
         Append (Buffer, Get_Name (Target));
      else
         Append (Buffer, Get_Name (Element));
      end if;
      
      --  Guard condition
      if Data.Has_Guard and Data.Guard_Length > 0 then
         Append (Buffer, " {");
         New_Line (Buffer);
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append (Buffer, "if ");
         Append (Buffer, Data.Guard_Expr (1 .. Data.Guard_Length));
         Append_Line (Buffer, ";");
         Append (Buffer, Ind);
         Append_Line (Buffer, "}");
      else
         Append_Line (Buffer, ";");
      end if;
      
      New_Line (Buffer);
   end Emit_Transition;

   --  ============================================================
   --  Requirement Emission
   --  ============================================================

   procedure Emit_Requirement (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
   begin
      Append (Buffer, Ind);
      Append (Buffer, "requirement def ");
      Append (Buffer, Get_Name (Element));
      Append_Line (Buffer, " {");
      
      if Options.Include_Comments then
         Append (Buffer, Ind);
         Append (Buffer, "    ");
         Append (Buffer, "doc /* Requirement: ");
         Append (Buffer, Get_Name (Element));
         Append_Line (Buffer, " */");
      end if;
      
      Append (Buffer, Ind);
      Append_Line (Buffer, "}");
      New_Line (Buffer);
   end Emit_Requirement;

   --  ============================================================
   --  Satisfy Emission
   --  ============================================================

   procedure Emit_Satisfy (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Options.Indent_Size);
      pragma Unreferenced (Options);
   begin
      Append (Buffer, Ind);
      Append (Buffer, "satisfy ");
      Append (Buffer, Get_Name (Element));
      Append_Line (Buffer, ";");
   end Emit_Satisfy;

   --  ============================================================
   --  Coverage Point Emission
   --  ============================================================

   procedure Emit_Coverage_Point (
      Point_ID   : in     String;
      Point_Type : in     String;
      Indent     : in     Natural;
      Buffer     : in Out Output_Buffer
   ) is
      Ind : constant String := Indent_String (Indent, Default_Indent);
   begin
      Append (Buffer, Ind);
      Append (Buffer, "// DO-331 Coverage Point: ");
      Append (Buffer, Point_ID);
      Append (Buffer, " (");
      Append (Buffer, Point_Type);
      Append_Line (Buffer, ")");
   end Emit_Coverage_Point;

end SysML_Emitter;
