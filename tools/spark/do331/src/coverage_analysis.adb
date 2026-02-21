--  STUNIR DO-331 Coverage Analysis Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Transformer_Utils; use Transformer_Utils;

package body Coverage_Analysis is

   --  ============================================================
   --  Buffer Operations
   --  ============================================================

   procedure Initialize_Report (Buffer : out Report_Buffer) is
   begin
      Buffer.Data := (others => ' ');
      Buffer.Length := 0;
   end Initialize_Report;

   procedure Append_Report (
      Buffer : in Out Report_Buffer;
      Text   : in     String
   ) is
   begin
      if Buffer.Length + Text'Length <= Max_Report_Length then
         Buffer.Data (Buffer.Length + 1 .. Buffer.Length + Text'Length) := Text;
         Buffer.Length := Buffer.Length + Text'Length;
      end if;
   end Append_Report;

   procedure Append_Line (
      Buffer : in Out Report_Buffer;
      Text   : in     String
   ) is
   begin
      Append_Report (Buffer, Text);
      Append_Report (Buffer, (1 => ASCII.LF));
   end Append_Line;

   function Get_Report_Content (Buffer : Report_Buffer) return String is
   begin
      if Buffer.Length > 0 then
         return Buffer.Data (1 .. Buffer.Length);
      else
         return "";
      end if;
   end Get_Report_Content;

   --  ============================================================
   --  Analysis Operations
   --  ============================================================

   function Compute_Stats (Points : Coverage_Points) return Coverage_Stats is
      Stats : Coverage_Stats := Null_Coverage_Stats;
   begin
      Stats.Total_Points := Points.Point_Count;
      
      for I in 1 .. Points.Point_Count loop
         --  Count by type
         case Points.Points (I).Point_Type is
            when State_Coverage      => Stats.State_Points := Stats.State_Points + 1;
            when Transition_Coverage => Stats.Transition_Points := Stats.Transition_Points + 1;
            when Decision_Coverage   => Stats.Decision_Points := Stats.Decision_Points + 1;
            when Condition_Coverage  => Stats.Condition_Points := Stats.Condition_Points + 1;
            when MCDC_Coverage       => Stats.MCDC_Points := Stats.MCDC_Points + 1;
            when Action_Coverage     => Stats.Action_Points := Stats.Action_Points + 1;
            when Entry_Coverage      => Stats.Entry_Points := Stats.Entry_Points + 1;
            when Exit_Coverage       => Stats.Exit_Points := Stats.Exit_Points + 1;
            when Path_Coverage       => Stats.Path_Points := Stats.Path_Points + 1;
            when Loop_Coverage       => Stats.Loop_Points := Stats.Loop_Points + 1;
         end case;
         
         --  Count instrumented and covered
         if Points.Points (I).Instrumented then
            Stats.Instrumented_Count := Stats.Instrumented_Count + 1;
         end if;
         
         if Points.Points (I).Covered then
            Stats.Covered_Count := Stats.Covered_Count + 1;
         end if;
      end loop;
      
      return Stats;
   end Compute_Stats;

   function Analyze (Points : Coverage_Points) return Analysis_Result is
      Result : Analysis_Result := Null_Analysis_Result;
   begin
      Result.Stats := Compute_Stats (Points);
      Result.Analysis_Time := Get_Current_Epoch;
      Result.Target_DAL := DAL_C;  -- Default
      
      --  Compute DAL coverage for each level
      for Level in DAL_Level loop
         declare
            DAL_Points : constant Coverage_Point_Array := Get_Points_For_DAL (Points, Level);
            Required   : Natural := 0;
            Achieved   : Natural := 0;
         begin
            Required := DAL_Points'Length;
            
            for I in DAL_Points'Range loop
               if DAL_Points (I).Covered then
                  Achieved := Achieved + 1;
               end if;
            end loop;
            
            Result.DAL_Coverage (Level).DAL := Level;
            Result.DAL_Coverage (Level).Required_Points := Required;
            Result.DAL_Coverage (Level).Achieved_Points := Achieved;
            
            if Required > 0 then
               Result.DAL_Coverage (Level).Coverage_Percent := (Achieved * 100) / Required;
            else
               Result.DAL_Coverage (Level).Coverage_Percent := 100;
            end if;
            
            Result.DAL_Coverage (Level).Meets_Objective := Achieved = Required;
         end;
      end loop;
      
      --  Check overall completeness
      Result.Is_Complete := Result.Stats.Covered_Count = Result.Stats.Total_Points;
      Result.Meets_Target := Result.DAL_Coverage (Result.Target_DAL).Meets_Objective;
      
      return Result;
   end Analyze;

   function Analyze_For_DAL (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return Analysis_Result is
      Result : Analysis_Result := Analyze (Points);
   begin
      Result.Target_DAL := Level;
      Result.Meets_Target := Result.DAL_Coverage (Level).Meets_Objective;
      return Result;
   end Analyze_For_DAL;

   function Meets_DAL_Requirements (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return Boolean is
      DAL_Points : constant Coverage_Point_Array := Get_Points_For_DAL (Points, Level);
   begin
      for I in DAL_Points'Range loop
         if not DAL_Points (I).Covered then
            return False;
         end if;
      end loop;
      return True;
   end Meets_DAL_Requirements;

   --  ============================================================
   --  Report Generation
   --  ============================================================

   procedure Generate_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Format  : in     Report_Format;
      Buffer  : in Out Report_Buffer
   ) is
   begin
      case Format is
         when JSON_Format =>
            Generate_JSON_Report (Points, Result, Buffer);
         when Text_Format =>
            Generate_Text_Report (Points, Result, Buffer);
         when HTML_Format =>
            Generate_HTML_Report (Points, Result, Buffer);
         when CSV_Format =>
            Generate_Text_Report (Points, Result, Buffer);  -- Fallback
      end case;
   end Generate_Report;

   procedure Generate_JSON_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   ) is
   begin
      Initialize_Report (Buffer);
      
      Append_Line (Buffer, "{");
      Append_Line (Buffer, "  \"schema\": \"stunir.coverage.do331.v1\",");
      Append_Report (Buffer, "  \"analysis_time\": ");
      Append_Report (Buffer, Natural_To_String (Result.Analysis_Time));
      Append_Line (Buffer, ",");
      
      --  Statistics
      Append_Line (Buffer, "  \"statistics\": {");
      Append_Report (Buffer, "    \"total_points\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Total_Points));
      Append_Line (Buffer, ",");
      Append_Report (Buffer, "    \"instrumented\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Instrumented_Count));
      Append_Line (Buffer, ",");
      Append_Report (Buffer, "    \"covered\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Covered_Count));
      Append_Line (Buffer, ",");
      Append_Report (Buffer, "    \"state_points\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.State_Points));
      Append_Line (Buffer, ",");
      Append_Report (Buffer, "    \"transition_points\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Transition_Points));
      Append_Line (Buffer, ",");
      Append_Report (Buffer, "    \"decision_points\": ");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Decision_Points));
      Append_Line (Buffer, "");
      Append_Line (Buffer, "  },");
      
      --  DAL Coverage
      Append_Line (Buffer, "  \"dal_coverage\": {");
      for Level in DAL_Level loop
         Append_Report (Buffer, "    \"");
         case Level is
            when DAL_A => Append_Report (Buffer, "DAL_A");
            when DAL_B => Append_Report (Buffer, "DAL_B");
            when DAL_C => Append_Report (Buffer, "DAL_C");
            when DAL_D => Append_Report (Buffer, "DAL_D");
            when DAL_E => Append_Report (Buffer, "DAL_E");
         end case;
         Append_Line (Buffer, "\": {");
         
         Append_Report (Buffer, "      \"required\": ");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Required_Points));
         Append_Line (Buffer, ",");
         
         Append_Report (Buffer, "      \"achieved\": ");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Achieved_Points));
         Append_Line (Buffer, ",");
         
         Append_Report (Buffer, "      \"percent\": ");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Coverage_Percent));
         Append_Line (Buffer, ",");
         
         Append_Report (Buffer, "      \"meets_objective\": ");
         if Result.DAL_Coverage (Level).Meets_Objective then
            Append_Report (Buffer, "true");
         else
            Append_Report (Buffer, "false");
         end if;
         Append_Line (Buffer, "");
         
         if Level = DAL_Level'Last then
            Append_Line (Buffer, "    }");
         else
            Append_Line (Buffer, "    },");
         end if;
      end loop;
      Append_Line (Buffer, "  },");
      
      --  Coverage points
      Append_Line (Buffer, "  \"coverage_points\": [");
      for I in 1 .. Points.Point_Count loop
         Append_Line (Buffer, "    {");
         Append_Report (Buffer, "      \"id\": \"");
         if Points.Points (I).Point_ID_Len > 0 then
            Append_Report (Buffer, Points.Points (I).Point_ID (1 .. Points.Points (I).Point_ID_Len));
         end if;
         Append_Line (Buffer, "\",");
         
         Append_Report (Buffer, "      \"type\": \"");
         Append_Report (Buffer, Get_Type_Prefix (Points.Points (I).Point_Type));
         Append_Line (Buffer, "\",");
         
         Append_Report (Buffer, "      \"instrumented\": ");
         if Points.Points (I).Instrumented then
            Append_Report (Buffer, "true");
         else
            Append_Report (Buffer, "false");
         end if;
         Append_Line (Buffer, ",");
         
         Append_Report (Buffer, "      \"covered\": ");
         if Points.Points (I).Covered then
            Append_Report (Buffer, "true");
         else
            Append_Report (Buffer, "false");
         end if;
         Append_Line (Buffer, "");
         
         if I < Points.Point_Count then
            Append_Line (Buffer, "    },");
         else
            Append_Line (Buffer, "    }");
         end if;
      end loop;
      Append_Line (Buffer, "  ]");
      
      Append_Line (Buffer, "}");
   end Generate_JSON_Report;

   procedure Generate_Text_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   ) is
      pragma Unreferenced (Points);
   begin
      Initialize_Report (Buffer);
      
      Append_Line (Buffer, "=" & "" & "========================================");
      Append_Line (Buffer, "STUNIR DO-331 Model Coverage Report");
      Append_Line (Buffer, "=" & "" & "========================================");
      Append_Line (Buffer, "");
      
      Append_Line (Buffer, "SUMMARY");
      Append_Line (Buffer, "-" & "" & "----------------------------------------");
      Append_Report (Buffer, "Total Coverage Points:  ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Total_Points));
      Append_Report (Buffer, "Instrumented Points:    ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Instrumented_Count));
      Append_Report (Buffer, "Covered Points:         ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Covered_Count));
      Append_Line (Buffer, "");
      
      Append_Line (Buffer, "COVERAGE BY TYPE");
      Append_Line (Buffer, "-" & "" & "----------------------------------------");
      Append_Report (Buffer, "State Coverage:       ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.State_Points));
      Append_Report (Buffer, "Transition Coverage:  ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Transition_Points));
      Append_Report (Buffer, "Decision Coverage:    ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Decision_Points));
      Append_Report (Buffer, "MC/DC Coverage:       ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.MCDC_Points));
      Append_Report (Buffer, "Entry/Exit Points:    ");
      Append_Line (Buffer, Natural_To_String (Result.Stats.Entry_Points + Result.Stats.Exit_Points));
      Append_Line (Buffer, "");
      
      Append_Line (Buffer, "DAL COMPLIANCE STATUS");
      Append_Line (Buffer, "-" & "" & "----------------------------------------");
      for Level in DAL_Level loop
         Append_Report (Buffer, "DAL ");
         case Level is
            when DAL_A => Append_Report (Buffer, "A");
            when DAL_B => Append_Report (Buffer, "B");
            when DAL_C => Append_Report (Buffer, "C");
            when DAL_D => Append_Report (Buffer, "D");
            when DAL_E => Append_Report (Buffer, "E");
         end case;
         Append_Report (Buffer, ": ");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Achieved_Points));
         Append_Report (Buffer, "/");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Required_Points));
         Append_Report (Buffer, " (");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Coverage_Percent));
         Append_Report (Buffer, "%) - ");
         if Result.DAL_Coverage (Level).Meets_Objective then
            Append_Line (Buffer, "PASS");
         else
            Append_Line (Buffer, "INCOMPLETE");
         end if;
      end loop;
      
      Append_Line (Buffer, "");
      Append_Line (Buffer, "=" & "" & "========================================");
      Append_Report (Buffer, "Overall Status: ");
      if Result.Meets_Target then
         Append_Line (Buffer, "COMPLIANT");
      else
         Append_Line (Buffer, "NON-COMPLIANT");
      end if;
      Append_Line (Buffer, "=" & "" & "========================================");
   end Generate_Text_Report;

   procedure Generate_HTML_Report (
      Points  : in     Coverage_Points;
      Result  : in     Analysis_Result;
      Buffer  : in Out Report_Buffer
   ) is
      pragma Unreferenced (Points);
   begin
      Initialize_Report (Buffer);
      
      Append_Line (Buffer, "<!DOCTYPE html>");
      Append_Line (Buffer, "<html><head><title>DO-331 Coverage Report</title>");
      Append_Line (Buffer, "<style>body{font-family:sans-serif;margin:20px}");
      Append_Line (Buffer, "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}");
      Append_Line (Buffer, ".pass{color:green}.fail{color:red}</style></head>");
      Append_Line (Buffer, "<body><h1>STUNIR DO-331 Model Coverage Report</h1>");
      
      Append_Line (Buffer, "<h2>Summary</h2>");
      Append_Line (Buffer, "<table>");
      Append_Report (Buffer, "<tr><td>Total Points</td><td>");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Total_Points));
      Append_Line (Buffer, "</td></tr>");
      Append_Report (Buffer, "<tr><td>Covered</td><td>");
      Append_Report (Buffer, Natural_To_String (Result.Stats.Covered_Count));
      Append_Line (Buffer, "</td></tr>");
      Append_Line (Buffer, "</table>");
      
      Append_Line (Buffer, "<h2>DAL Compliance</h2>");
      Append_Line (Buffer, "<table><tr><th>DAL</th><th>Required</th><th>Achieved</th><th>Status</th></tr>");
      for Level in DAL_Level loop
         Append_Report (Buffer, "<tr><td>DAL ");
         case Level is
            when DAL_A => Append_Report (Buffer, "A");
            when DAL_B => Append_Report (Buffer, "B");
            when DAL_C => Append_Report (Buffer, "C");
            when DAL_D => Append_Report (Buffer, "D");
            when DAL_E => Append_Report (Buffer, "E");
         end case;
         Append_Report (Buffer, "</td><td>");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Required_Points));
         Append_Report (Buffer, "</td><td>");
         Append_Report (Buffer, Natural_To_String (Result.DAL_Coverage (Level).Achieved_Points));
         Append_Report (Buffer, "</td><td class=\"");
         if Result.DAL_Coverage (Level).Meets_Objective then
            Append_Report (Buffer, "pass\">PASS");
         else
            Append_Report (Buffer, "fail\">INCOMPLETE");
         end if;
         Append_Line (Buffer, "</td></tr>");
      end loop;
      Append_Line (Buffer, "</table>");
      
      Append_Line (Buffer, "</body></html>");
   end Generate_HTML_Report;

   --  ============================================================
   --  DO-331 Compliance Helpers
   --  ============================================================

   function Get_Table_MBA5_Status (
      Points : Coverage_Points;
      Level  : DAL_Level
   ) return String is
   begin
      if Meets_DAL_Requirements (Points, Level) then
         return "COMPLIANT with DO-331 Table MB-A.5 for DAL " &
                (case Level is
                   when DAL_A => "A",
                   when DAL_B => "B",
                   when DAL_C => "C",
                   when DAL_D => "D",
                   when DAL_E => "E");
      else
         return "NON-COMPLIANT with DO-331 Table MB-A.5 for DAL " &
                (case Level is
                   when DAL_A => "A",
                   when DAL_B => "B",
                   when DAL_C => "C",
                   when DAL_D => "D",
                   when DAL_E => "E");
      end if;
   end Get_Table_MBA5_Status;

   function Check_Coverage_Objective (
      Points     : Coverage_Points;
      Level      : DAL_Level;
      Cov_Type   : Coverage_Type
   ) return Boolean is
   begin
      --  Check if coverage type is required for this DAL
      if not Is_Required (Level, Cov_Type) then
         return True;  -- Not required, so automatically passes
      end if;
      
      --  Check all points of this type are covered
      for I in 1 .. Points.Point_Count loop
         if Points.Points (I).Point_Type = Cov_Type and then
            Points.Points (I).DAL_Required (Level) and then
            not Points.Points (I).Covered
         then
            return False;
         end if;
      end loop;
      
      return True;
   end Check_Coverage_Objective;

end Coverage_Analysis;
