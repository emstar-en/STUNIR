--  STUNIR DO-331 Trace Matrix Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Transformer_Utils; use Transformer_Utils;

package body Trace_Matrix is

   --  ============================================================
   --  Buffer Operations
   --  ============================================================

   procedure Initialize_Export_Buffer (Buffer : out Export_Buffer) is
   begin
      Buffer.Data := (others => ' ');
      Buffer.Length := 0;
   end Initialize_Export_Buffer;

   procedure Append_Export (
      Buffer : in Out Export_Buffer;
      Text   : in     String
   ) is
   begin
      if Buffer.Length + Text'Length <= Max_Export_Length then
         Buffer.Data (Buffer.Length + 1 .. Buffer.Length + Text'Length) := Text;
         Buffer.Length := Buffer.Length + Text'Length;
      end if;
   end Append_Export;

   function Get_Export_Content (Buffer : Export_Buffer) return String is
   begin
      if Buffer.Length > 0 then
         return Buffer.Data (1 .. Buffer.Length);
      else
         return "";
      end if;
   end Get_Export_Content;

   --  Helper to append newline
   procedure Append_Line (
      Buffer : in Out Export_Buffer;
      Text   : in     String
   ) is
   begin
      Append_Export (Buffer, Text);
      Append_Export (Buffer, (1 => ASCII.LF));
   end Append_Line;

   --  ============================================================
   --  Export Operations
   --  ============================================================

   procedure Export_To_JSON (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   ) is
   begin
      Initialize_Export_Buffer (Buffer);
      
      Append_Line (Buffer, "{");
      Append_Line (Buffer, "  \"schema\": \"stunir.trace.do331.v1\",");
      
      --  Metadata
      Append_Export (Buffer, "  \"ir_hash\": \"");
      if Matrix.IR_Hash_Len > 0 then
         Append_Export (Buffer, Matrix.IR_Hash (1 .. Matrix.IR_Hash_Len));
      end if;
      Append_Line (Buffer, "\",");
      
      Append_Export (Buffer, "  \"model_hash\": \"");
      if Matrix.Model_Hash_Len > 0 then
         Append_Export (Buffer, Matrix.Model_Hash (1 .. Matrix.Model_Hash_Len));
      end if;
      Append_Line (Buffer, "\",");
      
      Append_Export (Buffer, "  \"created_at\": ");
      Append_Export (Buffer, Natural_To_String (Matrix.Created_At));
      Append_Line (Buffer, ",");
      
      Append_Export (Buffer, "  \"entry_count\": ");
      Append_Export (Buffer, Natural_To_String (Matrix.Entry_Count));
      Append_Line (Buffer, ",");
      
      --  Entries array
      Append_Line (Buffer, "  \"entries\": [");
      
      for I in 1 .. Matrix.Entry_Count loop
         Append_Line (Buffer, "    {");
         
         Append_Export (Buffer, "      \"trace_id\": ");
         Append_Export (Buffer, Natural_To_String (Matrix.Entries (I).Trace_ID));
         Append_Line (Buffer, ",");
         
         Append_Export (Buffer, "      \"source_id\": ");
         Append_Export (Buffer, Natural_To_String (Natural (Matrix.Entries (I).Source_ID mod 2**31)));
         Append_Line (Buffer, ",");
         
         Append_Export (Buffer, "      \"source_path\": \"");
         if Matrix.Entries (I).Source_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Source_Path (1 .. Matrix.Entries (I).Source_Path_Len));
         end if;
         Append_Line (Buffer, "\",");
         
         Append_Export (Buffer, "      \"target_id\": ");
         Append_Export (Buffer, Natural_To_String (Natural (Matrix.Entries (I).Target_ID mod 2**31)));
         Append_Line (Buffer, ",");
         
         Append_Export (Buffer, "      \"target_path\": \"");
         if Matrix.Entries (I).Target_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Target_Path (1 .. Matrix.Entries (I).Target_Path_Len));
         end if;
         Append_Line (Buffer, "\",");
         
         Append_Export (Buffer, "      \"rule\": \"");
         Append_Export (Buffer, IR_To_Model.Get_Rule_Name (Matrix.Entries (I).Rule));
         Append_Line (Buffer, "\",");
         
         Append_Export (Buffer, "      \"do331_objective\": \"");
         Append_Export (Buffer, IR_To_Model.Get_DO331_Objective (Matrix.Entries (I).Rule));
         Append_Line (Buffer, "\",");
         
         Append_Export (Buffer, "      \"verified\": ");
         if Matrix.Entries (I).Verified then
            Append_Export (Buffer, "true");
         else
            Append_Export (Buffer, "false");
         end if;
         Append_Line (Buffer, "");
         
         if I < Matrix.Entry_Count then
            Append_Line (Buffer, "    },");
         else
            Append_Line (Buffer, "    }");
         end if;
      end loop;
      
      Append_Line (Buffer, "  ]");
      Append_Line (Buffer, "}");
   end Export_To_JSON;

   procedure Export_To_CSV (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   ) is
   begin
      Initialize_Export_Buffer (Buffer);
      
      --  CSV header
      Append_Line (Buffer, "trace_id,source_id,source_path,target_id,target_path,rule,do331_objective,verified");
      
      --  Data rows
      for I in 1 .. Matrix.Entry_Count loop
         Append_Export (Buffer, Natural_To_String (Matrix.Entries (I).Trace_ID));
         Append_Export (Buffer, ",");
         Append_Export (Buffer, Natural_To_String (Natural (Matrix.Entries (I).Source_ID mod 2**31)));
         Append_Export (Buffer, ",");
         if Matrix.Entries (I).Source_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Source_Path (1 .. Matrix.Entries (I).Source_Path_Len));
         end if;
         Append_Export (Buffer, ",");
         Append_Export (Buffer, Natural_To_String (Natural (Matrix.Entries (I).Target_ID mod 2**31)));
         Append_Export (Buffer, ",");
         if Matrix.Entries (I).Target_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Target_Path (1 .. Matrix.Entries (I).Target_Path_Len));
         end if;
         Append_Export (Buffer, ",");
         Append_Export (Buffer, IR_To_Model.Get_Rule_Name (Matrix.Entries (I).Rule));
         Append_Export (Buffer, ",");
         Append_Export (Buffer, IR_To_Model.Get_DO331_Objective (Matrix.Entries (I).Rule));
         Append_Export (Buffer, ",");
         if Matrix.Entries (I).Verified then
            Append_Line (Buffer, "true");
         else
            Append_Line (Buffer, "false");
         end if;
      end loop;
   end Export_To_CSV;

   procedure Export_To_Text (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   ) is
   begin
      Initialize_Export_Buffer (Buffer);
      
      Append_Line (Buffer, "=" & "" & "=============================================");
      Append_Line (Buffer, "STUNIR DO-331 Traceability Matrix");
      Append_Line (Buffer, "=" & "" & "=============================================");
      Append_Line (Buffer, "");
      
      Append_Export (Buffer, "Total Entries: ");
      Append_Line (Buffer, Natural_To_String (Matrix.Entry_Count));
      
      Append_Export (Buffer, "Created At: ");
      Append_Line (Buffer, Natural_To_String (Matrix.Created_At));
      
      Append_Line (Buffer, "");
      Append_Line (Buffer, "Trace Entries:");
      Append_Line (Buffer, "-" & "" & "---------------------------------------------");
      
      for I in 1 .. Matrix.Entry_Count loop
         Append_Export (Buffer, "[" & Natural_To_String (Matrix.Entries (I).Trace_ID) & "] ");
         if Matrix.Entries (I).Source_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Source_Path (1 .. Matrix.Entries (I).Source_Path_Len));
         end if;
         Append_Export (Buffer, " -> ");
         if Matrix.Entries (I).Target_Path_Len > 0 then
            Append_Export (Buffer, Matrix.Entries (I).Target_Path (1 .. Matrix.Entries (I).Target_Path_Len));
         end if;
         Append_Line (Buffer, "");
         Append_Export (Buffer, "    Rule: ");
         Append_Export (Buffer, IR_To_Model.Get_Rule_Name (Matrix.Entries (I).Rule));
         Append_Export (Buffer, " (DO-331 ");
         Append_Export (Buffer, IR_To_Model.Get_DO331_Objective (Matrix.Entries (I).Rule));
         Append_Line (Buffer, ")");
      end loop;
   end Export_To_Text;

   --  ============================================================
   --  Summary Generation
   --  ============================================================

   function Get_Summary (
      Matrix : Traceability.Trace_Matrix
   ) return Matrix_Summary is
      Summary : Matrix_Summary;
      Forward_Count   : Natural := 0;
      Backward_Count  : Natural := 0;
      Verified_Count  : Natural := 0;
   begin
      Summary.Total_Entries := Matrix.Entry_Count;
      
      for I in 1 .. Matrix.Entry_Count loop
         case Matrix.Entries (I).Trace_Kind is
            when Trace_IR_To_Model =>
               Forward_Count := Forward_Count + 1;
            when Trace_Model_To_IR =>
               Backward_Count := Backward_Count + 1;
            when others =>
               null;
         end case;
         
         if Matrix.Entries (I).Verified then
            Verified_Count := Verified_Count + 1;
         end if;
      end loop;
      
      Summary.Forward_Traces := Forward_Count;
      Summary.Backward_Traces := Backward_Count;
      Summary.Verified_Count := Verified_Count;
      Summary.Unique_Sources := 0;  -- Would need set data structure
      Summary.Unique_Targets := 0;  -- Would need set data structure
      Summary.Rules_Used := 0;      -- Would need set data structure
      
      return Summary;
   end Get_Summary;

   --  ============================================================
   --  DO-331 Compliance Helpers
   --  ============================================================

   procedure Export_DO331_Table_MB1 (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   ) is
   begin
      Initialize_Export_Buffer (Buffer);
      
      Append_Line (Buffer, "DO-331 Table MB-1: Traceability Matrix");
      Append_Line (Buffer, "=======================================" & "");
      Append_Line (Buffer, "");
      Append_Line (Buffer, "| IR Element | Model Element | Transformation Rule | Verification Status |");
      Append_Line (Buffer, "|------------|---------------|---------------------|---------------------|");
      
      for I in 1 .. Matrix.Entry_Count loop
         Append_Export (Buffer, "| ");
         if Matrix.Entries (I).Source_Path_Len > 0 and Matrix.Entries (I).Source_Path_Len <= 20 then
            Append_Export (Buffer, Matrix.Entries (I).Source_Path (1 .. Matrix.Entries (I).Source_Path_Len));
         elsif Matrix.Entries (I).Source_Path_Len > 20 then
            Append_Export (Buffer, Matrix.Entries (I).Source_Path (1 .. 17));
            Append_Export (Buffer, "...");
         else
            Append_Export (Buffer, "N/A");
         end if;
         Append_Export (Buffer, " | ");
         
         if Matrix.Entries (I).Target_Path_Len > 0 and Matrix.Entries (I).Target_Path_Len <= 20 then
            Append_Export (Buffer, Matrix.Entries (I).Target_Path (1 .. Matrix.Entries (I).Target_Path_Len));
         elsif Matrix.Entries (I).Target_Path_Len > 20 then
            Append_Export (Buffer, Matrix.Entries (I).Target_Path (1 .. 17));
            Append_Export (Buffer, "...");
         else
            Append_Export (Buffer, "N/A");
         end if;
         Append_Export (Buffer, " | ");
         
         Append_Export (Buffer, IR_To_Model.Get_Rule_Name (Matrix.Entries (I).Rule));
         Append_Export (Buffer, " | ");
         
         if Matrix.Entries (I).Verified then
            Append_Export (Buffer, "Verified");
         else
            Append_Export (Buffer, "Pending");
         end if;
         Append_Line (Buffer, " |");
      end loop;
   end Export_DO331_Table_MB1;

   procedure Export_Bidirectional_Report (
      Matrix : in     Traceability.Trace_Matrix;
      Buffer : in Out Export_Buffer
   ) is
      Summary : constant Matrix_Summary := Get_Summary (Matrix);
   begin
      Initialize_Export_Buffer (Buffer);
      
      Append_Line (Buffer, "Bidirectional Traceability Report");
      Append_Line (Buffer, "=================================" & "");
      Append_Line (Buffer, "");
      
      Append_Export (Buffer, "Forward Traces (IR -> Model): ");
      Append_Line (Buffer, Natural_To_String (Summary.Forward_Traces));
      
      Append_Export (Buffer, "Backward Traces (Model -> IR): ");
      Append_Line (Buffer, Natural_To_String (Summary.Backward_Traces));
      
      Append_Export (Buffer, "Total Entries: ");
      Append_Line (Buffer, Natural_To_String (Summary.Total_Entries));
      
      Append_Export (Buffer, "Verified: ");
      Append_Line (Buffer, Natural_To_String (Summary.Verified_Count));
      
      if Summary.Total_Entries > 0 then
         Append_Export (Buffer, "Verification Rate: ");
         Append_Export (Buffer, Natural_To_String ((Summary.Verified_Count * 100) / Summary.Total_Entries));
         Append_Line (Buffer, "%");
      end if;
   end Export_Bidirectional_Report;

end Trace_Matrix;
