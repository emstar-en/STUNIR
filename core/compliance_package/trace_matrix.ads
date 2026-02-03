--  STUNIR Traceability Matrix Specification
--  Generates traceability matrices for DO-330 compliance
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Package_Types; use Package_Types;

package Trace_Matrix is

   Max_Output_Length : constant := 65536;

   subtype Output_Index is Positive range 1 .. Max_Output_Length;
   subtype Output_Length is Natural range 0 .. Max_Output_Length;
   subtype Output_Buffer is String (Output_Index);

   type Matrix_Format is (Text_Matrix, HTML_Matrix, CSV_Matrix, JSON_Matrix);

   --  Add traceability link to package
   procedure Add_Trace_Link
     (Comp_Pkg  : in out Compliance_Package;
      Source_ID : String;
      Target_ID : String;
      Kind      : Trace_Kind;
      Verified  : Boolean := False;
      Status    : out Package_Status)
   with Pre => Source_ID'Length > 0 and Source_ID'Length <= Max_Name_Length and
               Target_ID'Length > 0 and Target_ID'Length <= Max_Name_Length and
               Comp_Pkg.Trace_Total < Max_Trace_Entries,
        Post => (if Status = Success then
                  Comp_Pkg.Trace_Total = Comp_Pkg.Trace_Total'Old + 1);

   --  Verify all trace links
   function Verify_All_Traces (Comp_Pkg : Compliance_Package) return Boolean;

   --  Check if trace link exists
   function Trace_Exists
     (Comp_Pkg  : Compliance_Package;
      Source_ID : String;
      Target_ID : String) return Boolean
   with Pre => Source_ID'Length > 0 and Source_ID'Length <= Max_Name_Length and
               Target_ID'Length > 0 and Target_ID'Length <= Max_Name_Length;

   --  Generate traceability matrix
   procedure Generate_Matrix
     (Comp_Pkg : Compliance_Package;
      Format   : Matrix_Format;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status)
   with Pre => Comp_Pkg.Trace_Total >= 0,
        Post => (if Status = Success then Length > 0);

   --  Generate text format matrix
   procedure Generate_Text_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status);

   --  Generate HTML format matrix
   procedure Generate_HTML_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status);

   --  Generate CSV format matrix
   procedure Generate_CSV_Matrix
     (Comp_Pkg : Compliance_Package;
      Output   : out Output_Buffer;
      Length   : out Output_Length;
      Status   : out Package_Status);

   --  Calculate traceability coverage
   function Calculate_Trace_Coverage
     (Comp_Pkg : Compliance_Package) return Float;

   --  Check if package has complete traceability
   function Has_Complete_Traceability
     (Comp_Pkg : Compliance_Package) return Boolean;

   --  Count verified traces
   function Count_Verified_Traces
     (Comp_Pkg : Compliance_Package) return Natural;

end Trace_Matrix;
