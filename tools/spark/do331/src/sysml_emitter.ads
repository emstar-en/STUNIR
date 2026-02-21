--  STUNIR DO-331 SysML 2.0 Emitter Specification
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package generates SysML 2.0 textual notation from the Model IR.

pragma SPARK_Mode (On);

with Model_IR; use Model_IR;
with IR_To_Model; use IR_To_Model;

package SysML_Emitter is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Output_Length : constant := 1_000_000;  --  1 MB max output
   Max_Line_Width    : constant := 120;
   Default_Indent    : constant := 4;

   --  ============================================================
   --  Emitter Options
   --  ============================================================

   type Emitter_Options is record
      Indent_Size          : Positive := Default_Indent;
      Include_Comments     : Boolean := True;
      Include_Line_Numbers : Boolean := False;
      Include_Coverage     : Boolean := True;
      Include_Metadata     : Boolean := True;
      Max_Width            : Positive := Max_Line_Width;
   end record;

   Default_Emitter_Options : constant Emitter_Options := (
      Indent_Size          => Default_Indent,
      Include_Comments     => True,
      Include_Line_Numbers => False,
      Include_Coverage     => True,
      Include_Metadata     => True,
      Max_Width            => Max_Line_Width
   );

   --  ============================================================
   --  Emitter Result
   --  ============================================================

   type Emitter_Status is (
      Emit_Success,
      Emit_Partial,
      Emit_Buffer_Overflow,
      Emit_Invalid_Model,
      Emit_Error
   );

   type Emitter_Result is record
      Status        : Emitter_Status;
      Output_Length : Natural;
      Line_Count    : Natural;
      Element_Count : Natural;
   end record;

   --  ============================================================
   --  Output Buffer
   --  ============================================================

   type Output_Buffer is record
      Data       : String (1 .. Max_Output_Length);
      Length     : Natural := 0;
      Line_Count : Natural := 0;
   end record;

   --  Initialize buffer
   procedure Initialize_Buffer (Buffer : out Output_Buffer);

   --  Append string to buffer
   procedure Append (
      Buffer : in Out Output_Buffer;
      Text   : in     String
   );

   --  Append line to buffer (with newline)
   procedure Append_Line (
      Buffer : in Out Output_Buffer;
      Text   : in     String
   );

   --  Append newline
   procedure New_Line (Buffer : in out Output_Buffer);

   --  Get buffer content
   function Get_Content (Buffer : Output_Buffer) return String
     with Pre => Buffer.Length <= Max_Output_Length;

   --  ============================================================
   --  Main Emission Interface
   --  ============================================================

   --  Emit complete model to SysML 2.0
   procedure Emit_Model (
      Container : in     Model_Container;
      Storage   : in     Element_Storage;
      Options   : in     Emitter_Options;
      Buffer    : in Out Output_Buffer;
      Result    : out    Emitter_Result
   );

   --  ============================================================
   --  Element Emission
   --  ============================================================

   --  Emit package with contents
   procedure Emit_Package (
      Element : in     Model_Element;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit action definition
   procedure Emit_Action_Def (
      Element : in     Model_Element;
      Data    : in     Action_Data;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit attribute definition
   procedure Emit_Attribute (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit state definition
   procedure Emit_State (
      Element : in     Model_Element;
      Data    : in     State_Data;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit transition
   procedure Emit_Transition (
      Element : in     Model_Element;
      Data    : in     Transition_Data;
      Storage : in     Element_Storage;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit requirement definition
   procedure Emit_Requirement (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  Emit satisfy relationship
   procedure Emit_Satisfy (
      Element : in     Model_Element;
      Options : in     Emitter_Options;
      Indent  : in     Natural;
      Buffer  : in Out Output_Buffer
   );

   --  ============================================================
   --  Metadata Emission
   --  ============================================================

   --  Emit model header with metadata
   procedure Emit_Header (
      Container : in     Model_Container;
      Options   : in     Emitter_Options;
      Buffer    : in Out Output_Buffer
   );

   --  Emit coverage point comment
   procedure Emit_Coverage_Point (
      Point_ID   : in     String;
      Point_Type : in     String;
      Indent     : in     Natural;
      Buffer     : in Out Output_Buffer
   );

end SysML_Emitter;
