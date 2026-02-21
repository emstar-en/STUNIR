--  STUNIR DO-333 Certification Evidence Generator
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Verification_Condition; use Verification_Condition;

package body Evidence_Generator is

   --  ============================================================
   --  Helper: Append String
   --  ============================================================

   procedure Append
     (Content : in Out String;
      Length  : in Out Natural;
      Text    : String)
   with
      Pre => Length + Text'Length <= Content'Length
   is
   begin
      for I in Text'Range loop
         Length := Length + 1;
         Content (Content'First + Length - 1) := Text (I);
      end loop;
   end Append;

   --  ============================================================
   --  Helper: Append Natural
   --  ============================================================

   procedure Append_Natural
     (Content : in Out String;
      Length  : in Out Natural;
      Value   : Natural)
   with
      Pre => Length + 12 <= Content'Length  --  Enough for any Natural
   is
      Temp    : String (1 .. 12) := (others => ' ');
      Val     : Natural := Value;
      Idx     : Natural := 12;
   begin
      if Val = 0 then
         Temp (12) := '0';
      else
         while Val > 0 loop
            Temp (Idx) := Character'Val (Character'Pos ('0') + (Val mod 10));
            Val := Val / 10;
            Idx := Idx - 1;
         end loop;
      end if;

      for I in Temp'Range loop
         if Temp (I) /= ' ' then
            Append (Content, Length, Temp (I .. 12));
            return;
         end if;
      end loop;
   end Append_Natural;

   --  ============================================================
   --  Helper: New Line
   --  ============================================================

   procedure Append_Line
     (Content : in Out String;
      Length  : in Out Natural)
   with
      Pre => Length + 1 <= Content'Length
   is
   begin
      Length := Length + 1;
      Content (Content'First + Length - 1) := ASCII.LF;
   end Append_Line;

   --  ============================================================
   --  Generate PO Report
   --  ============================================================

   procedure Generate_PO_Report
     (PO_Coll  : PO_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   is
      Metrics : constant Coverage_Metrics := Get_Metrics (PO_Coll);
   begin
      Content := (others => ' ');
      Length := 0;

      case Format is
         when Format_Text =>
            Append (Content, Length, "DO-333 Proof Obligation Report");
            Append_Line (Content, Length);
            Append (Content, Length, "==============================");
            Append_Line (Content, Length);
            Append_Line (Content, Length);
            Append (Content, Length, "Total POs: ");
            Append_Natural (Content, Length, Metrics.Total_POs);
            Append_Line (Content, Length);
            Append (Content, Length, "Proved: ");
            Append_Natural (Content, Length, Metrics.Proved_POs);
            Append_Line (Content, Length);
            Append (Content, Length, "Unproved: ");
            Append_Natural (Content, Length, Metrics.Unproved_POs);
            Append_Line (Content, Length);
            Append (Content, Length, "Coverage: ");
            Append_Natural (Content, Length, Metrics.Coverage_Pct);
            Append (Content, Length, "%");
            Append_Line (Content, Length);

         when Format_JSON =>
            Append (Content, Length, "{");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""report_type"": ""proof_obligations"",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""total"": ");
            Append_Natural (Content, Length, Metrics.Total_POs);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""proved"": ");
            Append_Natural (Content, Length, Metrics.Proved_POs);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""unproved"": ");
            Append_Natural (Content, Length, Metrics.Unproved_POs);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""coverage_percent"": ");
            Append_Natural (Content, Length, Metrics.Coverage_Pct);
            Append_Line (Content, Length);
            Append (Content, Length, "}");
            Append_Line (Content, Length);

         when Format_HTML =>
            Append (Content, Length, "<html><head><title>PO Report</title></head>");
            Append_Line (Content, Length);
            Append (Content, Length, "<body><h1>Proof Obligation Report</h1>");
            Append_Line (Content, Length);
            Append (Content, Length, "<p>Total: ");
            Append_Natural (Content, Length, Metrics.Total_POs);
            Append (Content, Length, "</p>");
            Append_Line (Content, Length);
            Append (Content, Length, "<p>Proved: ");
            Append_Natural (Content, Length, Metrics.Proved_POs);
            Append (Content, Length, "</p>");
            Append_Line (Content, Length);
            Append (Content, Length, "</body></html>");

         when Format_XML =>
            Append (Content, Length, "<?xml version=""1.0""?>");
            Append_Line (Content, Length);
            Append (Content, Length, "<po_report>");
            Append_Line (Content, Length);
            Append (Content, Length, "  <total>");
            Append_Natural (Content, Length, Metrics.Total_POs);
            Append (Content, Length, "</total>");
            Append_Line (Content, Length);
            Append (Content, Length, "</po_report>");

         when Format_CSV =>
            Append (Content, Length, "Type,Total,Proved,Unproved,Coverage");
            Append_Line (Content, Length);
            Append (Content, Length, "PO,");
            Append_Natural (Content, Length, Metrics.Total_POs);
            Append (Content, Length, ",");
            Append_Natural (Content, Length, Metrics.Proved_POs);
            Append (Content, Length, ",");
            Append_Natural (Content, Length, Metrics.Unproved_POs);
            Append (Content, Length, ",");
            Append_Natural (Content, Length, Metrics.Coverage_Pct);
            Append_Line (Content, Length);
      end case;

      Result := Gen_Success;
   end Generate_PO_Report;

   --  ============================================================
   --  Generate VC Report
   --  ============================================================

   procedure Generate_VC_Report
     (VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   is
      Coverage : constant VC_Coverage_Report := Get_Coverage (VC_Coll);
   begin
      Content := (others => ' ');
      Length := 0;

      case Format is
         when Format_Text =>
            Append (Content, Length, "DO-333 Verification Condition Report");
            Append_Line (Content, Length);
            Append (Content, Length, "=====================================");
            Append_Line (Content, Length);
            Append_Line (Content, Length);
            Append (Content, Length, "Total VCs: ");
            Append_Natural (Content, Length, Coverage.Total_VCs);
            Append_Line (Content, Length);
            Append (Content, Length, "Valid: ");
            Append_Natural (Content, Length, Coverage.Valid_VCs);
            Append_Line (Content, Length);
            Append (Content, Length, "Invalid: ");
            Append_Natural (Content, Length, Coverage.Invalid_VCs);
            Append_Line (Content, Length);
            Append (Content, Length, "Coverage: ");
            Append_Natural (Content, Length, Coverage.Coverage_Pct);
            Append (Content, Length, "%");
            Append_Line (Content, Length);

         when Format_JSON =>
            Append (Content, Length, "{");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""report_type"": ""verification_conditions"",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""total"": ");
            Append_Natural (Content, Length, Coverage.Total_VCs);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""valid"": ");
            Append_Natural (Content, Length, Coverage.Valid_VCs);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""coverage_percent"": ");
            Append_Natural (Content, Length, Coverage.Coverage_Pct);
            Append_Line (Content, Length);
            Append (Content, Length, "}");
            Append_Line (Content, Length);

         when others =>
            Append (Content, Length, "VC Report");
            Append_Line (Content, Length);
      end case;

      Result := Gen_Success;
   end Generate_VC_Report;

   --  ============================================================
   --  Generate Coverage Report
   --  ============================================================

   procedure Generate_Coverage_Report
     (PO_Coll  : PO_Collection;
      VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   is
      PO_Metrics : constant Coverage_Metrics := Get_Metrics (PO_Coll);
      VC_Coverage : constant VC_Coverage_Report := Get_Coverage (VC_Coll);
   begin
      Content := (others => ' ');
      Length := 0;

      case Format is
         when Format_Text =>
            Append (Content, Length, "DO-333 Coverage Report");
            Append_Line (Content, Length);
            Append (Content, Length, "======================");
            Append_Line (Content, Length);
            Append_Line (Content, Length);
            Append (Content, Length, "Proof Obligations:");
            Append_Line (Content, Length);
            Append (Content, Length, "  Total: ");
            Append_Natural (Content, Length, PO_Metrics.Total_POs);
            Append_Line (Content, Length);
            Append (Content, Length, "  Coverage: ");
            Append_Natural (Content, Length, PO_Metrics.Coverage_Pct);
            Append (Content, Length, "%");
            Append_Line (Content, Length);
            Append_Line (Content, Length);
            Append (Content, Length, "Verification Conditions:");
            Append_Line (Content, Length);
            Append (Content, Length, "  Total: ");
            Append_Natural (Content, Length, VC_Coverage.Total_VCs);
            Append_Line (Content, Length);
            Append (Content, Length, "  Coverage: ");
            Append_Natural (Content, Length, VC_Coverage.Coverage_Pct);
            Append (Content, Length, "%");
            Append_Line (Content, Length);

         when Format_JSON =>
            Append (Content, Length, "{");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""report_type"": ""coverage"",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""po_coverage"": ");
            Append_Natural (Content, Length, PO_Metrics.Coverage_Pct);
            Append (Content, Length, ",");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""vc_coverage"": ");
            Append_Natural (Content, Length, VC_Coverage.Coverage_Pct);
            Append_Line (Content, Length);
            Append (Content, Length, "}");
            Append_Line (Content, Length);

         when others =>
            Append (Content, Length, "Coverage Report");
            Append_Line (Content, Length);
      end case;

      Result := Gen_Success;
   end Generate_Coverage_Report;

   --  ============================================================
   --  Initialize Matrix
   --  ============================================================

   procedure Initialize_Matrix
     (Matrix : out Compliance_Matrix)
   is
      procedure Add_Entry
        (M     : in Out Compliance_Matrix;
         Obj   : String;
         Desc  : String)
      is
         E : Compliance_Entry := Empty_Compliance_Entry;
      begin
         if M.Count >= Max_Compliance_Entries then
            return;
         end if;

         for I in Obj'Range loop
            E.Objective_ID (I - Obj'First + 1) := Obj (I);
         end loop;
         E.Obj_Len := Obj'Length;

         for I in Desc'Range loop
            exit when I - Desc'First + 1 > Max_Desc_Len;
            E.Description (I - Desc'First + 1) := Desc (I);
         end loop;
         E.Desc_Len := Natural'Min (Desc'Length, Max_Desc_Len);
         E.Status := Status_Not_Applicable;

         M.Count := M.Count + 1;
         M.Entries (M.Count) := E;
      end Add_Entry;
   begin
      Matrix := Empty_Matrix;

      --  DO-333 objectives
      Add_Entry (Matrix, "FM.1", "Formal Specification");
      Add_Entry (Matrix, "FM.2", "Formal Verification");
      Add_Entry (Matrix, "FM.3", "Proof Coverage");
      Add_Entry (Matrix, "FM.4", "VC Management");
      Add_Entry (Matrix, "FM.5", "FM Integration");
      Add_Entry (Matrix, "FM.6", "Certification Evidence");
   end Initialize_Matrix;

   --  ============================================================
   --  Update Matrix Entry
   --  ============================================================

   procedure Update_Matrix_Entry
     (Matrix      : in Out Compliance_Matrix;
      Objective   : String;
      Status      : Compliance_Status;
      Evidence    : String;
      Comment     : String)
   is
   begin
      for I in 1 .. Matrix.Count loop
         declare
            E : Compliance_Entry renames Matrix.Entries (I);
         begin
            if E.Obj_Len = Objective'Length then
               declare
                  Match : Boolean := True;
               begin
                  for J in Objective'Range loop
                     if E.Objective_ID (J - Objective'First + 1) /= Objective (J) then
                        Match := False;
                        exit;
                     end if;
                  end loop;

                  if Match then
                     E.Status := Status;

                     for K in Evidence'Range loop
                        exit when K - Evidence'First + 1 > Max_Evidence_Len;
                        E.Evidence_Ref (K - Evidence'First + 1) := Evidence (K);
                     end loop;
                     E.Evid_Len := Natural'Min (Evidence'Length, Max_Evidence_Len);

                     for K in Comment'Range loop
                        exit when K - Comment'First + 1 > Max_Comment_Len;
                        E.Comments (K - Comment'First + 1) := Comment (K);
                     end loop;
                     E.Comment_Len := Natural'Min (Comment'Length, Max_Comment_Len);

                     return;
                  end if;
               end;
            end if;
         end;
      end loop;
   end Update_Matrix_Entry;

   --  ============================================================
   --  Generate Compliance Matrix
   --  ============================================================

   procedure Generate_Compliance_Matrix
     (PO_Coll  : PO_Collection;
      VC_Coll  : VC_Collection;
      Format   : Output_Format;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   is
      pragma Unreferenced (PO_Coll, VC_Coll);
      Matrix : Compliance_Matrix;
   begin
      Content := (others => ' ');
      Length := 0;

      Initialize_Matrix (Matrix);

      case Format is
         when Format_Text =>
            Append (Content, Length, "DO-333 Compliance Matrix");
            Append_Line (Content, Length);
            Append (Content, Length, "========================");
            Append_Line (Content, Length);
            Append_Line (Content, Length);

            for I in 1 .. Matrix.Count loop
               Append (Content, Length,
                 Matrix.Entries (I).Objective_ID (1 .. Matrix.Entries (I).Obj_Len));
               Append (Content, Length, ": ");
               Append (Content, Length,
                 Matrix.Entries (I).Description (1 .. Matrix.Entries (I).Desc_Len));
               Append (Content, Length, " - ");
               Append (Content, Length, Status_Name (Matrix.Entries (I).Status));
               Append_Line (Content, Length);
            end loop;

         when Format_JSON =>
            Append (Content, Length, "{");
            Append_Line (Content, Length);
            Append (Content, Length, "  ""compliance_matrix"": [");
            Append_Line (Content, Length);

            for I in 1 .. Matrix.Count loop
               Append (Content, Length, "    {");
               Append (Content, Length, """objective"": """);
               Append (Content, Length,
                 Matrix.Entries (I).Objective_ID (1 .. Matrix.Entries (I).Obj_Len));
               Append (Content, Length, """, ");
               Append (Content, Length, """status"": """);
               Append (Content, Length, Status_Name (Matrix.Entries (I).Status));
               Append (Content, Length, """}");
               if I < Matrix.Count then
                  Append (Content, Length, ",");
               end if;
               Append_Line (Content, Length);
            end loop;

            Append (Content, Length, "  ]");
            Append_Line (Content, Length);
            Append (Content, Length, "}");
            Append_Line (Content, Length);

         when others =>
            Append (Content, Length, "Compliance Matrix");
            Append_Line (Content, Length);
      end case;

      Result := Gen_Success;
   end Generate_Compliance_Matrix;

   --  ============================================================
   --  Generate Justification Template
   --  ============================================================

   procedure Generate_Justification_Template
     (PO_Coll  : PO_Collection;
      Content  : out String;
      Length   : out Natural;
      Result   : out Generation_Result)
   is
      Analysis : constant Unproven_Analysis := Get_Unproven_Analysis (PO_Coll);
   begin
      Content := (others => ' ');
      Length := 0;

      Append (Content, Length, "DO-333 Unproven VC Justification Template");
      Append_Line (Content, Length);
      Append (Content, Length, "==========================================");
      Append_Line (Content, Length);
      Append_Line (Content, Length);

      Append (Content, Length, "Critical Unproven: ");
      Append_Natural (Content, Length, Analysis.Critical_Unproven);
      Append_Line (Content, Length);
      Append (Content, Length, "Safety Unproven: ");
      Append_Natural (Content, Length, Analysis.Safety_Unproven);
      Append_Line (Content, Length);
      Append_Line (Content, Length);

      Append (Content, Length, "For each unproven PO, provide:");
      Append_Line (Content, Length);
      Append (Content, Length, "1. PO ID and location");
      Append_Line (Content, Length);
      Append (Content, Length, "2. Reason proof failed");
      Append_Line (Content, Length);
      Append (Content, Length, "3. Alternative evidence (review/test)");
      Append_Line (Content, Length);
      Append (Content, Length, "4. Risk assessment");
      Append_Line (Content, Length);
      Append (Content, Length, "5. Reviewer signature and date");
      Append_Line (Content, Length);

      Result := Gen_Success;
   end Generate_Justification_Template;

   --  ============================================================
   --  Format Name
   --  ============================================================

   function Format_Name (F : Output_Format) return String is
   begin
      case F is
         when Format_Text => return "Text";
         when Format_HTML => return "HTML";
         when Format_JSON => return "JSON";
         when Format_XML  => return "XML";
         when Format_CSV  => return "CSV";
      end case;
   end Format_Name;

   --  ============================================================
   --  Format Extension
   --  ============================================================

   function Format_Extension (F : Output_Format) return String is
   begin
      case F is
         when Format_Text => return ".txt";
         when Format_HTML => return ".html";
         when Format_JSON => return ".json";
         when Format_XML  => return ".xml";
         when Format_CSV  => return ".csv";
      end case;
   end Format_Extension;

   --  ============================================================
   --  Status Name
   --  ============================================================

   function Status_Name (S : Compliance_Status) return String is
   begin
      case S is
         when Status_Compliant      => return "Compliant";
         when Status_Partial        => return "Partial";
         when Status_Non_Compliant  => return "Non-Compliant";
         when Status_Not_Applicable => return "N/A";
      end case;
   end Status_Name;

end Evidence_Generator;
