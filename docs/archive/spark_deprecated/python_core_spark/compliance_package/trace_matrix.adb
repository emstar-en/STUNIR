--  STUNIR Traceability Matrix Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Trace_Matrix is

   procedure Copy_String
     (Source : String;
      Target : out String;
      Length : out Natural)
   with Pre => Target'Length >= Source'Length
   is
   begin
      Length := Source'Length;
      for I in 1 .. Source'Length loop
         pragma Loop_Invariant (I <= Source'Length);
         Target (Target'First + I - 1) := Source (Source'First + I - 1);
      end loop;
      for I in Source'Length + 1 .. Target'Length loop
         Target (Target'First + I - 1) := ' ';
      end loop;
   end Copy_String;

   procedure Append_To_Buffer
     (Buffer   : in out Output_Buffer;
      Position : in out Output_Length;
      Text     : String;
      Status   : out Package_Status)
   is
   begin
      if Position + Text'Length > Max_Output_Length then
         Status := IO_Error;
         return;
      end if;
      for I in Text'Range loop
         pragma Loop_Invariant (Position + (I - Text'First) < Max_Output_Length);
         Buffer (Position + 1 + (I - Text'First)) := Text (I);
      end loop;
      Position := Position + Text'Length;
      Status := Success;
   end Append_To_Buffer;

   procedure Add_Trace_Link
     (Comp_Pkg  : in out Compliance_Package;
      Source_ID : String;
      Target_ID : String;
      Kind      : Trace_Kind;
      Verified  : Boolean := False;
      Status    : out Package_Status)
   is
      T : Trace_Entry := Null_Trace_Entry;
      Src_Len, Tgt_Len : Natural;
   begin
      Copy_String (Source_ID, T.Source_ID, Src_Len);
      T.Source_Len := Package_Types.Name_Length (Src_Len);
      Copy_String (Target_ID, T.Target_ID, Tgt_Len);
      T.Target_Len := Package_Types.Name_Length (Tgt_Len);
      T.Kind := Kind;
      T.Verified := Verified;
      T.Is_Valid := True;
      Comp_Pkg.Trace_Total := Comp_Pkg.Trace_Total + 1;
      Comp_Pkg.Traces (Comp_Pkg.Trace_Total) := T;
      Status := Success;
   end Add_Trace_Link;

   function Verify_All_Traces (Comp_Pkg : Compliance_Package) return Boolean is
   begin
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (I <= Comp_Pkg.Trace_Total);
         if not Comp_Pkg.Traces (I).Verified then
            return False;
         end if;
      end loop;
      return True;
   end Verify_All_Traces;

   function Trace_Exists
     (Comp_Pkg  : Compliance_Package;
      Source_ID : String;
      Target_ID : String) return Boolean
   is
   begin
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (I <= Comp_Pkg.Trace_Total);
         declare
            T : Trace_Entry renames Comp_Pkg.Traces (I);
         begin
            if T.Source_Len = Source_ID'Length and
               T.Target_Len = Target_ID'Length
            then
               declare
                  Src_Match : Boolean := True;
                  Tgt_Match : Boolean := True;
               begin
                  for J in 1 .. Source_ID'Length loop
                     pragma Loop_Invariant (J <= Source_ID'Length);
                     if T.Source_ID (J) /= Source_ID (Source_ID'First + J - 1)
                     then
                        Src_Match := False;
                        exit;
                     end if;
                  end loop;
                  for J in 1 .. Target_ID'Length loop
                     pragma Loop_Invariant (J <= Target_ID'Length);
                     if T.Target_ID (J) /= Target_ID (Target_ID'First + J - 1)
                     then
                        Tgt_Match := False;
                        exit;
                     end if;
                  end loop;
                  if Src_Match and Tgt_Match then
                     return True;
                  end if;
               end;
            end if;
         end;
      end loop;
      return False;
   end Trace_Exists;

   procedure Generate_Matrix
     (Comp_Pkg : Compliance_Package;
      Format   : Matrix_Format;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status)
   is
   begin
      case Format is
         when Text_Matrix =>
            Generate_Text_Matrix (Comp_Pkg, Output, Length, Status);
         when HTML_Matrix =>
            Generate_HTML_Matrix (Comp_Pkg, Output, Length, Status);
         when CSV_Matrix =>
            Generate_CSV_Matrix (Comp_Pkg, Output, Length, Status);
         when JSON_Matrix =>
            Generate_Text_Matrix (Comp_Pkg, Output, Length, Status);
      end case;
   end Generate_Matrix;

   procedure Generate_Text_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status)
   is
      Pos : Output_Length := 0;
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos, "Traceability Matrix" & (1 => ASCII.LF), Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, "==================" & (1 => ASCII.LF), Status);
      if Status /= Success then return; end if;
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            T : Trace_Entry renames Comp_Pkg.Traces (I);
         begin
            Append_To_Buffer (Output, Pos, T.Source_ID (1 .. T.Source_Len),
                              Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, " -> ", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, T.Target_ID (1 .. T.Target_Len),
                              Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, (1 => ASCII.LF), Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Length := Pos;
   end Generate_Text_Matrix;

   procedure Generate_HTML_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status)
   is
      Pos : Output_Length := 0;
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos,
         "<!DOCTYPE html><html><head><title>Trace Matrix</title></head>",
         Status);
      if Status /= Success then return; end if;
      Append_To_Buffer (Output, Pos, "<body><h1>Traceability</h1><table>",
                        Status);
      if Status /= Success then return; end if;
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            T : Trace_Entry renames Comp_Pkg.Traces (I);
         begin
            Append_To_Buffer (Output, Pos, "<tr><td>", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, T.Source_ID (1 .. T.Source_Len),
                              Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, "</td><td>", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, T.Target_ID (1 .. T.Target_Len),
                              Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, "</td></tr>", Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Append_To_Buffer (Output, Pos, "</table></body></html>", Status);
      Length := Pos;
   end Generate_HTML_Matrix;

   procedure Generate_CSV_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status)
   is
      Pos : Output_Length := 0;
   begin
      Output := (others => ' ');
      Length := 0;
      Status := Success;
      Append_To_Buffer (Output, Pos, "Source,Target,Verified" & (1 => ASCII.LF),
                        Status);
      if Status /= Success then return; end if;
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (Pos <= Max_Output_Length);
         declare
            T : Trace_Entry renames Comp_Pkg.Traces (I);
         begin
            Append_To_Buffer (Output, Pos, T.Source_ID (1 .. T.Source_Len),
                              Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, ",", Status);
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, T.Target_ID (1 .. T.Target_Len),
                              Status);
            if Status /= Success then return; end if;
            if T.Verified then
               Append_To_Buffer (Output, Pos, ",Yes", Status);
            else
               Append_To_Buffer (Output, Pos, ",No", Status);
            end if;
            if Status /= Success then return; end if;
            Append_To_Buffer (Output, Pos, (1 => ASCII.LF), Status);
            if Status /= Success then return; end if;
         end;
      end loop;
      Length := Pos;
   end Generate_CSV_Matrix;

   function Calculate_Trace_Coverage
     (Comp_Pkg : Compliance_Package) return Float
   is
      Verified_Count : Natural := 0;
   begin
      if Comp_Pkg.Trace_Total = 0 then
         return 0.0;
      end if;
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (Verified_Count <= I - 1);
         if Comp_Pkg.Traces (I).Verified then
            Verified_Count := Verified_Count + 1;
         end if;
      end loop;
      return Float (Verified_Count) / Float (Comp_Pkg.Trace_Total) * 100.0;
   end Calculate_Trace_Coverage;

   function Has_Complete_Traceability
     (Comp_Pkg : Compliance_Package) return Boolean
   is
   begin
      return Comp_Pkg.Trace_Total > 0 and Verify_All_Traces (Comp_Pkg);
   end Has_Complete_Traceability;

   function Count_Verified_Traces
     (Comp_Pkg : Compliance_Package) return Natural
   is
      Count : Natural := 0;
   begin
      for I in 1 .. Comp_Pkg.Trace_Total loop
         pragma Loop_Invariant (Count <= I - 1);
         if Comp_Pkg.Traces (I).Verified then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Verified_Traces;

end Trace_Matrix;
